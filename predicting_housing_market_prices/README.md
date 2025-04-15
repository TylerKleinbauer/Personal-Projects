## What Drives Swiss House Prices?
Using Lasso regression (test R² = 0.83), I identified 12 key predictors from 165 features. The model reveals:

- **Momentum Rules**: Last year’s house and apartment asking prices (`houses_asking_price_lag1`, coefficient 12.6292) dominate—Swiss markets are sticky, with buyers anchoring to recent trends.
- **Apartments Lead**: Apartment price growth and transactions (`appartments_asking_growth`, `appartments_transaction_growth`) spill into houses—urban demand ripples outward.
- **Migration Matters**: Net migration (`ch_net_migration_lag1`, 0.9583) drives competition—newcomers fuel housing shortages.
- **Money Moves**: Currency and M3 growth (`currency_in_circulation_lag1`, 0.8691) reflect liquidity—cheap loans post-2015 pushed prices up. Inflation’s small lag effect (-0.0277) hints at borrowing ease.
- **Investment Echoes**: Apartment building transactions (`lag2`, 1.4544) suggest investor shifts to houses, though early data gaps (1970-1983) limit certainty.
- **Banking Caveat**: Foreign loans (`utilisation_lag2`, 0.3308) appear predictive but were static—likely a minor player.

**Data Note**: Some features (e.g., apartment transactions, loans) were backfilled pre-1984 due to missing records. While the model performs well (CV R² = 0.88), I interpret these cautiously—post-1984 data drives reliability.