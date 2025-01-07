import torch
import numpy as np
from collections import defaultdict
from model import PathCon
from utils import sparse_to_tuple


args = None


def train(model_args, data):
    global args, model, sess
    args = model_args

    # extract data
    triplets, paths, n_relations, neighbor_params, path_params = data

    train_triplets, valid_triplets, test_triplets = triplets
    train_edges = torch.LongTensor(np.array(range(len(train_triplets)), np.int32))
    # 训练集有边

    train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32))
    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))
    # 训练，验证，测试对应的实体对集合

    train_paths, valid_paths, test_paths = paths

    train_labels = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    # define the model
    model = PathCon(args, n_relations, neighbor_params, path_params)
    # 构建模型

    optimizer = torch.optim.Adam(
        # Adam 优化算法，一种基于梯度下降的算法，能够自动调节学习率
        filter(lambda p: p.requires_grad, model.parameters()), 
        # requires_grad 属性指示该参数是否需要计算梯度，只有需要梯度的参数才会被包括在优化器中
        lr=args.lr,
        # lr=args.lr 设置优化器的学习率，这里的 args.lr 是从命令行参数或配置文件中获取的学习率值。

        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    true_relations = defaultdict(set)
    # 对于每个实体对 (head, tail)，true_relations 将包含所有与之相关的真实关系。
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # 用来等着装算出来的acc, mrr, mr, hit1, hit3, hit5

    print('start training ...')

    for step in range(args.epoch):

        # shuffle training data
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        # 打乱索引的顺序
        if args.use_context:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]
        if args.use_path:
            train_paths = train_paths[index]
        train_labels = train_labels[index]

        # training
        s = 0
        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_paths, train_labels, s, s + args.batch_size))
            s += args.batch_size

        # evaluation
        print('epoch %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_paths, train_labels)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_paths, valid_labels)
        test_acc, test_scores = evaluate(test_entity_pairs, test_paths, test_labels)

        # show evaluation result for current epoch
        current_res = 'acc: %.4f' % test_acc
        print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5)
        print('           mrr: %.4f   mr: %.4f   h1: %.4f   h3: %.4f   h5: %.4f' % (mrr, mr, hit1, hit3, hit5))
        print()

        # update final results according to validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    # show final evaluation result
    print('final results\n%s' % final_res)


def get_feed_dict(entity_pairs, train_edges, paths, labels, start, end):
    # 从完整的数据集中提取相应的实体对、路径和标签，并将它们转换为适合模型输入的格式。
    feed_dict = {}

    if args.use_context:
        feed_dict["entity_pairs"] = entity_pairs[start:end]
        if train_edges is not None:
            feed_dict["train_edges"] = train_edges[start:end]
        else:
            # for evaluation no edges should be masked out
            # 如果 train_edges 为空，即在评估阶段，不需要屏蔽任何边缘。
            # 因此，创建一个填充了 -1 的数组，长度为当前批次的大小（end - start），
            # 并将其转换为 torch.LongTensor。
            feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
                        else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    if args.use_path:
        if args.path_type == 'embedding':
            indices, values, shape = sparse_to_tuple(paths[start:end])
            #sparse_to_tuple 的目的是将一个稀疏矩阵（以任何格式存储）转换为 COO（Coordinate）格式的元组表示。
            # 这种表示形式包括三个主要组件：非零元素索引、非零元素值、矩阵形状。

            indices = torch.LongTensor(indices).cuda() if args.cuda else torch.LongTensor(indices)
            values = torch.Tensor(values).cuda() if args.cuda else torch.Tensor(values)
            # torch.LongTensor 用于存储整数索引，而 torch.Tensor 用于存储实数值。

            feed_dict["path_features"] = torch.sparse.FloatTensor(indices.t(), values, torch.Size(shape)).to_dense()
            # 这行代码使用 indices、values 和 shape 创建一个稀疏的 FloatTensor。indices.t() 是转置索引矩阵，
            # 以适应 PyTorch 稀疏张量的要求。然后，使用 to_dense() 方法将稀疏张量转换为密集张量，
            # 这样就可以在模型中方便地使用。这个密集张量被添加到 feed_dict 中，以供模型使用。
        elif args.path_type == 'rnn':
            feed_dict["path_ids"] = torch.LongTensor(paths[start:end]).cuda() if args.cuda \
                    else torch.LongTensor(paths[start:end])

    feed_dict["labels"] = labels[start:end]

    return feed_dict


def evaluate(entity_pairs, paths, labels):
    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.test_step(model, get_feed_dict(
            entity_pairs, None, paths, labels, s, s + args.batch_size))
        acc_list.extend(acc)
        scores_list.extend(scores)
        s += args.batch_size

    return float(np.mean(acc_list)), np.array(scores_list)


def calculate_ranking_metrics(triplets, scores, true_relations):
    # scores：模型为每个三元组生成的预测得分，形状为 [batch_size, n_relations]。
    # true_relations：一个二维数组，表示每个实体对的真实关系集合。

    for i in range(scores.shape[0]):
        # 遍历每个三元组，对于每个实体对，
        # 从它们的真实关系集合中减去已知的关系，并对预测得分进行调整，使得真实关系的得分降低。
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1) 
    # 对预测得分按每个样本的维度进行降序(-score)排序，axis=1说明是横着对行排序，得到每个样本的排名索引。

    relations = np.array(triplets)[0:scores.shape[0], 2]
    # np.array(triplets) 将 triplets 转换为 NumPy 数组，以便可以使用 NumPy 的索引功能。
    # [0:scores.shape[0], 2] 选择 triplets 数组的前 scores.shape[0] 行（
    # 即与 scores 数组相同数量的样本）和第三列（索引为 2，代表关系）。
    # 结果 relations 是一个一维数组，包含每个样本的真实关系索引。

    sorted_indices -= np.expand_dims(relations, 1)
    # 假设：
    #sorted_indices = [[1, 2, 3], [4, 5, 6]]
    #relations = [1, 2]
    #执行 np.expand_dims(relations, 1) 后，relations 变为 [[1], [2]]。
    #执行 sorted_indices -= np.expand_dims(relations, 1) 后，操作如下：
    #对于第一行：[1, 2, 3] - [1] = [0, 1, 2]
    #对于第二行：[4, 5, 6] - [2] = [2, 3, 4]

    zero_coordinates = np.argwhere(sorted_indices == 0)
    # 找出为0的元素的序列，也就是对应真实关系的序列

    rankings = zero_coordinates[:, 1] + 1
    #因为索引从0开始，所以加1得到实际的排名

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5
