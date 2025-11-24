# Maddpg Microgrid Best

**Training Date**: 2025-11-24 11:01:42

## 📊 Model Performance

### Overall Performance
- **Total Reward**: 30039.03 cents ≈ $300.39
- **Episodes Trained**: 200
- **Learning Progress**: 524.9% improvement
- **Best Reward**: 51468.09
- **Final Average**: 30109.81

### Agent Performance Breakdown

| Agent | Reward (cents) | Reward ($) | Contribution |
|-------|----------------|------------|--------------|
| Wind     |           0.00 | $      0.00 |         0.0% |
| Solar    |        2411.55 | $     24.12 |         8.0% |
| Diesel   |       28039.67 | $    280.40 |        93.3% |
| Battery  |          38.65 | $      0.39 |         0.1% |
| Customer |        -450.84 | $     -4.51 |        -1.5% |

### Performance Ratings

**Overall Rating**: ⚠️  Good (20k-40k)

- 🌬️ **Wind**: ⚠️ Low Profit (0.00 cents)
- ☀️ **Solar**: ⚠️ Low Profit (2411.55 cents)
- ⛽ **Diesel**: ✅ Profitable (28039.67 cents)
- 🔋 **Battery**: ✅ Positive (38.65 cents)
- 👥 **Customer**: ✅ Cost Savings (-450.84 cents)

## ⚙️  Training Configuration

### Hyperparameters
```json
{
  "actor_lr": N/A,
  "critic_lr": N/A,
  "gamma": N/A,
  "tau": N/A,
  "batch_size": N/A,
  "buffer_capacity": N/A
}
```

## 🚀 Quick Start

See `usage_example.py` for complete testing code.

---
**Generated**: 2025-11-24 11:01:42