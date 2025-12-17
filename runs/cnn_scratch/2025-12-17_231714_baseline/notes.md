# Run: baseline

Key hyperparams:
- epochs=80, batch_size=64, lr=0.001, optimizer=adam, loss=bce_logits
- activation=relu, base_channels=32, blocks=[2, 2, 2, 2]
- early_stopping={'enabled': True, 'monitor': 'val_loss', 'patience': 10, 'min_delta': 0.0005}
- device=mps
