# Stock Market Simulation Based on Investor Behavior

## Project Introduction

This is a stock market simulation system based on behavioral finance implemented in Python. The project aims to simulate the behavior of different types of investors through computer modeling, observe how they influence stock market price fluctuations and trading volumes, and track the asset changes of various investor types.

This project originated from a simple idea: since stock market fluctuations mainly come from the behaviors of different types of participants (investors), can we simulate the stock market by modeling these participants' behaviors through computer simulation? With the development of AIGC technology, this idea has been realized and formed into this complete simulation system.

## Features

- **Multiple Investor Types**: Simulates various investors with different behavioral characteristics, including value investors, trend investors, momentum traders, random investors, and more
- **Market Trading Mechanism**: Uses a call auction method to simulate daily trading, generating price series and trading volume series
- **Visualization Analysis**: Provides visual displays of stock price and trading volume fluctuations, as well as analysis of asset changes for each type of investor
- **Transaction Fee Simulation**: Implements buying and selling commission mechanisms
- **Capital Injection and Withdrawal**: Supports simulation of capital injection and withdrawal functions

## Investor Types

The project simulates the following types of investors:

1. **Value Investor (ValueInvestor)**:
2. **Trend Investor (TrendInvestor)**:
3. **Momentum Trader (ChaseInvestor)**:
4. **Random Investor (RandomInvestor)**:
5. **Never Stop Loss Investor (NeverStopLossInvestor)**:
6. **Bottom Fishing Investor (BottomFishingInvestor)**:
7. **Message Investor (MessageInvestor)**:

## Market Mechanism

- Uses a single call auction to simulate each day's trading
- Generates price series and trading volume series
- Implements transaction fee mechanisms (buy rate and sell rate)
- Supports capital injection and withdrawal functions
- Tracks executed trading volumes and historical records

## Visualization Output

The simulation results output includes:
- Stock price and trading volume fluctuations
- Average changes in assets, cash, and positions for each type of investor
- Marking of capital change events on the price chart

## How to Use

### Requirements

- Python 3.6+
- NumPy
- Matplotlib

### Running the Simulation

1. Clone or download this project to your local machine
2. Run the latest version of the Python script:

```bash
python main_ca_3.2.1.1.py
```

3. Observe the generated simulation results and charts

### Customizing the Simulation

You can customize the simulation by modifying parameters in the script:

- Adjust the number and initial capital of different types of investors
- Modify the number of simulation days
- Adjust transaction fee rates
- Set capital injection or withdrawal events

## Version History

### Version 3.2.1.1

- Enhanced investor behavior modeling:
  - Added Bottom Fishing Investor class, implementing batch buying strategies
  - Added Message Investor class with delayed information processing capabilities
  - Removed Insider Trader class to maintain market fairness

- Improved market mechanisms:
  - Added transaction fee mechanisms (buy rate and sell rate)
  - Implemented capital injection and withdrawal functions
  - Added executed trading volume tracking and historical records

- Enhanced visualization:
  - Added trading volume display in the first subplot
  - Added capital change event markers on the price chart
  - Improved trend investor performance analysis through MA periods

- Code optimization:
  - Improved error handling and parameter validation
  - Enhanced documentation and code comments
  - Better separation of concerns in market operations

## Project Structure

- `main_ca_3.*.py`: Various versions of the code
- `StockGame_CN.pdf`: Chinese version of the project development process and research analysis document, which contains some interesting findings
- `StockGame.pdf`: English version of the project development process and research analysis document (translated from StockGame_CN.pdf using Google Translate)
- `StockGame_CN.html`: Chinese HTML version of the project development process and research analysis document (same content as StockGame_CN.pdf)
- `StockGame.htm`: English HTML version of the project development process and research analysis document (same content as StockGame.pdf)
- `README.md`: Project documentation
- `README_CN.md`: Chinese version of project documentation
- `LICENSE`: License file

## Notes

- This project is for educational and research purposes only and does not constitute any investment advice
- Simulation results do not represent real market behavior and are for reference only
- Investor types and behaviors are simplified models; real market investor behaviors are more complex

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Copyright

Copyright (c) 2023 Stock Market Simulation Game. All rights reserved.

This software and associated documentation files are protected by copyright law. Unauthorized reproduction or distribution of this software, or any portion of it, may result in severe civil and criminal penalties.