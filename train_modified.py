import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

import torch
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.hparams import HParam
from utils import metrics
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from utils.logger import MyWriter

# Import your custom dataset
from dataset.kvasir_dataset import get_datasets  # adjust this if your file is named differently

def main(hp, num_epochs, resume, name):
    checkpoint_dir = os.path.join(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(hp.log, name)
    os.makedirs(log_dir, exist_ok=True)
    writer = MyWriter(log_dir)

    # model selection
    model = ResUnetPlusPlus(3).cuda() if hp.RESNET_PLUS_PLUS else ResUnet(3, 64).cuda()

    # loss and optimizer
    criterion = metrics.BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # resume logic
    best_loss = 999
    start_epoch = 0
    if resume and os.path.isfile(resume):
        print(f"=> loading checkpoint '{resume}'")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        print(f"=> loaded checkpoint '{resume}' (epoch {checkpoint['epoch']})")
    elif resume:
        print(f"=> no checkpoint found at '{resume}'")

    # dataset and dataloaders
    train_dataset, val_dataset = get_datasets(
        image_dir=hp.train + "/input",
        mask_dir=hp.train + "/output",
        val_size=0.2,
        seed=42
    )

    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    step = 0
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n{'-' * 10}")
        lr_scheduler.step()

        train_acc = metrics.MetricTracker()
        train_loss = metrics.MetricTracker()

        loader = tqdm(train_loader, desc="training")
        for idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            train_loss.update(loss.item(), outputs.size(0))

            if step % hp.logging_step == 0:
                writer.log_training(train_loss.avg, train_acc.avg, step)
                loader.set_description(f"Training Loss: {train_loss.avg:.4f} Acc: {train_acc.avg:.4f}")

            if step % hp.validation_interval == 0:
                valid_metrics = validation(val_loader, model, criterion, writer, step)
                save_path = os.path.join(checkpoint_dir, f"{name}_checkpoint_{step:04d}.pt")
                best_loss = min(valid_metrics["valid_loss"], best_loss)
                torch.save({
                    "step": step,
                    "epoch": epoch,
                    "arch": "ResUnetPlusPlus" if hp.RESNET_PLUS_PLUS else "ResUnet",
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                print(f"Saved checkpoint to: {save_path}")

            step += 1


def validation(valid_loader, model, criterion, logger, step):
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(valid_loader, desc="validation")):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
            valid_loss.update(loss.item(), outputs.size(0))

            if idx == 0:
                logger.log_images(inputs.cpu(), labels.cpu(), outputs.cpu(), step)

    logger.log_validation(valid_loss.avg, valid_acc.avg, step)
    print(f"Validation Loss: {valid_loss.avg:.4f} Acc: {valid_acc.avg:.4f}")
    model.train()

    return {"valid_loss": valid_loss.avg, "valid_acc": valid_acc.avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polyp Segmentation Training")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML config file")
    parser.add_argument("--epochs", default=75, type=int, help="Number of epochs to run")
    parser.add_argument("--resume", default="", type=str, help="Checkpoint path")
    parser.add_argument("--name", default="default", type=str, help="Experiment name")

    args = parser.parse_args()

    hp = HParam(args.config)
    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name)
