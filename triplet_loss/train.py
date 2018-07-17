from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from sklearn.metrics.pairwise import pairwise_distances

from loss.triplet import triplet_margin_loss
from mnist.net import MNIST_NET
from mnist.visual import VisdomLinePlotter
from mnist.data import origin_data, triplet_data


net = MNIST_NET(classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=1e-5)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
best_accuracy = 0.0

vis = VisdomLinePlotter(env_name='visual')


def train(model, epoch, train_sets, train_loader):
    global vis
    model.train()
    losses = 0.0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item() * targets.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
    losses = losses / len(train_sets)
    accuracy = correct / len(train_sets)
    print('Train epoch {}, loss: {:.4f}, accuracy: {:.4f}'.format(epoch, losses, accuracy))
    vis.plot('loss', 'train_loss', epoch_idx+1, losses)
    vis.plot('accuracy', 'train_acc', epoch_idx+1, accuracy)

def tests(model, epoch, tests_sets, tests_loader):
    global best_accuracy, vis
    model.eval()
    losses = 0.0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(tests_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            losses += loss.item() * targets.size(0)
            _, preds= torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
    losses = losses / len(tests_sets)
    accuracy = correct / len(tests_sets)
    print("Test epoch {}, loss: {:.4f}, accuracy: {:.4f}".format(epoch, losses, accuracy))
    vis.plot('loss', 'tests_loss', epoch_idx+1, losses)
    vis.plot('accuracy', 'tests_acc', epoch_idx+1, accuracy)

    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        state = {
            'accuracy': best_accuracy,
            'epoch': epoch+1,
            'net_state': model.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/%s.t7' % 'MNIST_NET')


###########################################################################################
#########################Train MNIST with Triplet Loss#####################################

triplet_net = MNIST_NET()
margin = 1.0
triplet_criterion = nn.TripletMarginLoss(margin=margin)
triplet_optimizer = optim.Adam(triplet_net.parameters(), weight_decay=1e-5)
triplet_net.to(device)

def train_with_triplet_loss(model, epoch, train_sets, triplet_train_loader):
    model.train()
    losses = 0.0
    for batch_idx, (anc, pos, neg) in enumerate(triplet_train_loader):
        anc = anc.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        anchor = model(anc)
        positive = model(pos)
        negative = model(neg)
        loss = triplet_criterion(anchor, positive, negative)

        losses += loss.item() * anc.size(0)

        triplet_optimizer.zero_grad()
        loss.backward()
        triplet_optimizer.step()

    losses /= len(train_sets)
    print('Train epoch {}, loss: {:.4f}'.format(epoch, losses))
    vis.plot('triplet loss', 'train_loss', epoch, losses)

def tests_with_triplet_loss(model, epoch, tests_sets, triplet_tests_loader):
    model.eval()
    losses = 0.0
    for batch_idx, (anc, pos, neg) in enumerate(triplet_tests_loader):
        anc = anc.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        with torch.set_grad_enabled(False):
            anchor = model(anc)
            positive = model(pos)
            negative = model(neg)
            loss = triplet_criterion(anchor, positive, negative)

            losses += loss.item() * anc.size(0)

    losses /= len(tests_sets)
    print('Test epoch {}, loss: {:.4f}'.format(epoch, losses))
    vis.plot('triplet loss', 'test_loss', epoch, losses)


###########################################################################################
##################Trian MNIST with Triplet Loss Using Batch Hard Method####################

def batch_hard(embedings, targets):
    triplets = []
    dis = pairwise_distances(embedings)
    for idx in range(targets.shape[0]):
        spec_idxes_eq = np.where(targets == targets[idx])[0]
        spec_idxes_ne = np.where(targets != targets[idx])[0]
        # Hardest positive example
        hardest_pos_idx = -1
        hardest_pos_dis = 0.0
        for pos_idx in range(spec_idxes_eq.shape[0]):
            index = spec_idxes_eq[pos_idx]
            if dis[idx][index] == 0.0:
                continue
            if hardest_pos_dis < dis[idx][index]:
                hardest_pos_dis = dis[idx][index]
                hardest_pos_idx = index
        if hardest_pos_idx == -1:
            continue
        # Hardest negative example
        hardest_neg_idx = -1
        hardest_neg_dis = 100000.0
        for neg_idx in range(spec_idxes_ne.shape[0]):
            index = spec_idxes_ne[neg_idx]
            if hardest_neg_dis > dis[idx][index]:
                hardest_neg_dis = dis[idx][index]
                hardest_neg_idx = index
        if hardest_neg_idx == -1:
            continue
        triplets.append((idx, hardest_pos_idx, hardest_neg_idx))
    return triplets

def batch_hard_triplets(embedings, triplets):
    anchors = embedings[triplets[0][0]].unsqueeze(0)
    positives = embedings[triplets[0][1]].unsqueeze(0)
    negatives = embedings[triplets[0][2]].unsqueeze(0)
    for idx in range(1, len(triplets)):
        anchors = torch.cat([anchors, embedings[triplets[idx][0]].unsqueeze(0)], 0)
        positives = torch.cat([positives, embedings[triplets[idx][1]].unsqueeze(0)], 0)
        negatives = torch.cat([negatives, embedings[triplets[idx][2]].unsqueeze(0)], 0)
    return anchors, positives, negatives

def train_with_batch_hard(model, epoch, train_sets, train_loader):
    model.train()
    losses = 0.0
    num_examples = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        embedings = model(inputs)

        embedings_detached = embedings.detach()
        triplets = batch_hard(embedings_detached.data.cpu().numpy(), targets.numpy())
        if len(triplets) == 0:
            continue

        anchors, positives, negatives = batch_hard_triplets(embedings, triplets)

        loss = triplet_criterion(anchors, positives, negatives)
        num_examples += anchors.size(0)
        losses += loss.item() * anchors.size(0)

        triplet_optimizer.zero_grad()
        loss.backward()
        triplet_optimizer.step()

    losses /= num_examples
    print('Train epoch {}, loss: {:.4f}'.format(epoch, losses))
    vis.plot('batch hard triplet loss', 'train_loss', epoch, losses)

def tests_with_batch_hard(model, epoch, tests_sets, tests_loader):
    model.eval()
    losses = 0.0
    num_examples = 0
    for batch_idx, (inputs, targets) in enumerate(tests_loader):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            embedings = model(inputs)

            embedings_detached = embedings.detach()
            triplets = batch_hard(embedings_detached.data.cpu().numpy(), targets.numpy())
            if len(triplets) == 0:
                continue

            anchors, positives, negatives = batch_hard_triplets(embedings, triplets)

            loss = triplet_criterion(anchors, positives, negatives)
            num_examples += anchors.size(0)
            losses += loss.item() * anchors.size(0)

    losses /= num_examples
    print('Tests epoch {}, loss: {:.4f}'.format(epoch, losses))
    vis.plot('batch hard triplet loss', 'test_loss', epoch, losses)


if __name__ == '__main__':
    # train_sets, tests_sets, train_loader, tests_loader = origin_data()
    # for epoch_idx in range(10):
    #    train(net, epoch_idx, train_sets, train_loader)
    #    tests(net, epoch_idx, tests_sets, tests_loader)

    # train_sets, tests_sets, triplet_train_loader, triplet_tests_loader = triplet_data()
    # for epoch_idx in range(10):
    #    train_with_triplet_loss(triplet_net, epoch_idx, train_sets, triplet_train_loader)
    #    tests_with_triplet_loss(triplet_net, epoch_idx, tests_sets, triplet_tests_loader)

    train_sets, tests_sets, train_loader, tests_loader = origin_data()
    for epoch_idx in range(10):
        train_with_batch_hard(triplet_net, epoch_idx, train_sets, train_loader)
        tests_with_batch_hard(triplet_net, epoch_idx, tests_sets, train_loader)
