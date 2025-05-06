# @title TokenEconomyModel
import mesa
import random
from mesa.datacollection import DataCollector
from Miner_Agent import StorageMiner
from Token_Holders import TokenHolder
from Market_Maker import MarketMaker

class TokenEconomyModel(mesa.Model):
    def __init__(self, params):
        super().__init__()
        self.verbose = getattr(params, "verbose", False)
        self.token_price = params.initial_token_price
        self.monthly_token_distribution = getattr(params, "monthly_token_distribution", 1000000)
        self.simulation_months = params.simulation_months
        self.schedule = mesa.time.RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                "TokenPrice": lambda m: m.token_price,
                "ActiveMiners": lambda m: len(m.active_miners),
                "TotalNetworkStorage": self.get_total_active_storage,
                "CirculatingSupply": lambda m: sum(a.tokens for a in m.schedule.agents),
            },
            agent_reporters={
                "Tokens": "tokens",
                "FiatBalance": "fiat_balance",
                "Type": lambda a: getattr(a, "agent_type", "unknown"),
            },
        )

        self.transaction_history = []
        self.price_history = []
        self.running = True
        self.max_steps = self.simulation_months
        self.active_miners = []
        self.inactive_miners = []

        # Instantiate StorageMiners
        for i in range(params.initial_miners):
            miner = StorageMiner(
                unique_id=i,
                model=self,
                storage=random.uniform(1.0, 5.0),
                efficiency=random.uniform(0.8, 1.2),
                operating_cost=params.operating_cost_per_miner,
                initial_fiat=params.initial_fiat_per_miner
            )
            self.schedule.add(miner)
            self.active_miners.append(miner)

        # Instantiate TokenHolders
        for i, strategy in enumerate(params.holder_strategies):
            holder = TokenHolder(
                unique_id=1000 + i,
                model=self,
                investment_strategy=strategy,
                risk_tolerance=random.uniform(0.3, 0.7),
                price_belief=params.initial_token_price
            )
            self.schedule.add(holder)

        # Instantiate a single MarketMaker
        market_maker = MarketMaker(
            unique_id=9999,
            model=self,
            liquidity_fiat=params.market_maker_liquidity,
            liquidity_tokens=params.market_maker_liquidity / params.initial_token_price,
            fee_rate=params.market_fee_rate,
            price_volatility=params.market_price_volatility
        )
        self.market_maker = market_maker
        self.schedule.add(market_maker)

    def get_total_active_storage(self):
        return sum(miner.storage for miner in self.active_miners if miner.active)

    def record_transaction(self, tx_type, amount, value, agent_id):
        self.transaction_history.append({
            "type": tx_type,
            "amount": amount,
            "value": value,
            "agent_id": agent_id,
        })

    def step(self):
        if self.verbose:
            print(f"Step {self.schedule.steps}: Token Price = {self.token_price:.4f}, Active Miners = {len(self.active_miners)}")
        self.datacollector.collect(self)
        self.token_price = self.market_maker.provide_price_quote()
        self.price_history.append(self.token_price)
        self.schedule.step()
        if self.schedule.steps >= self.max_steps:
            self.running = False
