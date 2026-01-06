# Run: opt_sgd_lr1e-3

Key hyperparams:
- epochs=80, batch_size=64, lr=0.001, optimizer=sgd, loss=bce_logits
- activation=relu, base_channels=32, blocks=[2, 2, 2, 2]
- early_stopping={'enabled': True, 'monitor': 'val_loss', 'patience': 10, 'min_delta': 0.0005}
- device=mps

Config sources:
{
  "base": "/Users/timschafer/Documents/Masterstudium/2. Semester/Ku\u0308nstliche Intelligenz/QR_Code_Windshield/qr-code-windshield-cnn/configs/cnn/baseline.yaml",
  "overrides": [
    "/Users/timschafer/Documents/Masterstudium/2. Semester/Ku\u0308nstliche Intelligenz/QR_Code_Windshield/qr-code-windshield-cnn/configs/cnn/exp_opt_sgd.yaml"
  ]
}
