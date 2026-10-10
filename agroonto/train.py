"""
Training utilities for AgroOnto-VLM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from . import config


def ontology_label_smoothing(labels, onto_sim, num_classes,
                             lambda_ce, lambda_uniform, lambda_onto):
    """Build soft target distribution from ontology similarity matrix."""
    batch_size = labels.size(0)
    one_hot = F.one_hot(labels, num_classes).float()
    uniform = torch.ones_like(one_hot) / num_classes
    onto_dist = onto_sim[labels]

    soft_targets = lambda_ce * one_hot + lambda_uniform * uniform + lambda_onto * onto_dist
    soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
    return soft_targets


def train_one_epoch(model, train_loader, optimizer, scaler, onto_sim,
                    num_classes, lambda_ce, lambda_uniform, lambda_onto,
                    lambda_proto, device):
    """Train for one epoch with ontology-guided hybrid loss."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            logits_cls, logits_clip, features = model(images)

            soft_targets = ontology_label_smoothing(
                labels, onto_sim, num_classes,
                lambda_ce, lambda_uniform, lambda_onto)

            log_probs = F.log_softmax(logits_cls, dim=-1)
            loss_cls = -(soft_targets * log_probs).sum(dim=-1).mean()

            loss_clip = F.cross_entropy(logits_clip, labels)

            loss = 0.5 * loss_cls + 0.5 * loss_clip

            if lambda_proto > 0:
                loss = loss + lambda_proto * _proto_loss(features, labels, num_classes)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]],
            config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        preds = logits_cls.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, 100.0 * correct / total


def _proto_loss(features, labels, num_classes):
    """Prototype contrastive loss."""
    protos = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            protos.append(features[mask].mean(dim=0))
        else:
            protos.append(torch.zeros_like(features[0]))
    protos = torch.stack(protos)
    protos = protos / protos.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    sim = features @ protos.T
    return F.cross_entropy(sim * 10, labels)


def evaluate(model, val_loader, device):
    """Evaluate model accuracy on validation set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                logits_cls, _, _ = model(images)
            preds = logits_cls.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total
