# References

Full bibliography for the FX fix-aware market-making project. The four
load-bearing references are highlighted in the main README; this document
collects the broader background literature.

## FX Fix Pattern

- **Krohn, I., Mueller, P. & Whelan, P.** (2024). *Foreign Exchange Fixings and Returns Around the Clock.* Journal of Finance, 79(1), 541–578. — Replicated empirically in Section 6/7 of the demo: pre-fix USD appreciation, post-fix reversal.
- **Evans, M. D. D. & Lyons, R. K.** (2008). *How is macro news transmitted to exchange rates?* Journal of Financial Economics, 88, 26–50. — FX market microstructure foundations: order flow as the carrier between fundamentals and prices.
- **Cartea, Á. & Sánchez-Betancourt, L.** (2023). *Toxic Flow Detection in FX Markets.* — Adverse-selection regimes around fix windows from trade-flow imbalance signals; frames the pre-fix vs at-fix markout split.
- **Melvin, M. & Prins, J.** (2015). *Equity hedging and exchange rates at the London 4 p.m. fix.* Journal of International Financial Markets, Institutions & Money, 31, 50–69. — The asymmetric-hedging mechanism: foreign holders of US assets hedge USD; US holders of foreign assets typically don't.

## Curve Construction & Pricing

- **Hagan, P. S. & West, G.** (2006). *Interpolation Methods for Curve Construction.* Applied Mathematical Finance, 13(2), 89–129. — Monotone convex interpolation; the method used in `interpolation.py`.
- **Henrard, M.** (2014). *Interest Rate Modelling in the Multi-Curve Framework.* Palgrave Macmillan. — Post-crisis multi-curve framework, basis spreads, OIS discounting, meeting-date curve construction.
- **Andersen, L. & Piterbarg, V.** (2010). *Interest Rate Modeling*, Vols I–III. Atlantic Financial Press.
- **Ametrano, F. & Bianchetti, M.** (2009). *Bootstrapping the Illiquidity: Multiple Yield Curves Construction for Market Coherent Forward Rates Estimation.* — Multi-curve bootstrap framework post-GFC.

## Cross-Currency Basis

- **Du, W., Tepper, A. & Verdelhan, A.** (2018). *Deviations from Covered Interest Rate Parity.* Journal of Finance, 73(3), 915–957. — Structural negative-basis story for EUR/USD and JPY/USD.
- **Borio, C., McCauley, R., McGuire, P. & Sushko, V.** (2016). *Covered interest parity lost: understanding the cross-currency basis.* BIS Quarterly Review.
- **Avdjiev, S., Du, W., Koch, C. & Shin, H. S.** (2019). *The Dollar, Bank Leverage and Deviations from Covered Interest Parity.* AER: Insights. — The balance-sheet mechanism behind the year-end TOY widening.

## Execution & Market Microstructure

- **Almgren, R. & Chriss, N.** (2001). *Optimal Execution of Portfolio Transactions.* Journal of Risk, 3(2), 5–39. — Permanent + temporary impact; implemented in `market_impact.py`.
- **Almgren, R.** (2003). *Optimal Execution with Nonlinear Impact Functions and Trading-Enhanced Risk.* Applied Mathematical Finance, 10(1), 1–18.
- **Cartea, Á., Jaimungal, S. & Penalva, J.** (2015). *Algorithmic and High-Frequency Trading.* Cambridge University Press. — The E[PnL] = P(hit)·(margin − cost) framework used in the quote optimizer.

## Quoting & Win Probability

- **Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X.** (2013). *Applied Logistic Regression*, 3rd ed. Wiley. — Decile calibration and goodness-of-fit; used in `win_probability.py`.
- **Avellaneda, M. & Stoikov, S.** (2008). *High-Frequency Trading in a Limit Order Book.* Quantitative Finance, 8(3), 217–224.
- **Barzykin, A., Bergault, P. & Guéant, O.** (2021–2023). *Multi-Currency Optimal Market Making.* — HSBC framework for joint quoting across correlated FX pairs with inventory constraints.

## Industry Reference

- **Bank for International Settlements** (2025). *Triennial Central Bank Survey of Foreign Exchange and OTC Derivatives Markets.* — Global FX turnover ~$9.6T/day.
