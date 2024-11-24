import random


# Let's start with a super simple version
class SimpleAgent:
    def __init__(self, agent_type):
        self.inventory = {"wood": 0, "stone": 0, "hammer": 0}
        # Carpenter can craft but has limited storage
        self.is_carpenter = agent_type == "carpenter"
        self.hammer_capacity = 1 if self.is_carpenter else 10

    def can_craft_hammer(self):
        return (
            self.is_carpenter
            and self.inventory["wood"] >= 1
            and self.inventory["stone"] >= 1
            and self.inventory["hammer"] < self.hammer_capacity
        )


class EnhancedEconomicAdaSociety:

    def __init__(self):
        self.agents = {
            "carpenter_1": SimpleAgent("carpenter"),
            "carpenter_2": SimpleAgent("carpenter"),
            "miner_1": SimpleAgent("miner"),
            "miner_2": SimpleAgent("miner"),
        }

        # Add stats tracking
        self.stats = {
            "gdp": 0,
            "total_trades": 0,
            "total_crafted": 0,
            "market_prices": []
        }

        # Market conditions with more economic factors
        self.market = {
            "base_prices": {"wood": 1, "stone": 1, "hammer": 5},
            "resource_availability": {
                "wood": 100,  # Total in environment
                "stone": 100,
                "replenish_rate": 2,  # Per round
            },
            "supply_chain": {
                "efficiency": 1.0,
                "disruption_chance": 0.1,
                "recovery_rate": 0.2,
            },
        }

    def run_competition(self, rounds=20):
        trade_history = {
            "carpenter_1": {
                "profit": 0,
                "trades": 0,
                "reputation": 1.0,
                "resources_gathered": 0,
            },
            "carpenter_2": {
                "profit": 0,
                "trades": 0,
                "reputation": 1.0,
                "resources_gathered": 0,
            },
        }

        for round in range(rounds):
            print(f"\n=== Round {round + 1} ===")

            # Resource replenishment phase
            self._replenish_resources()

            # Resource gathering phase
            self._handle_resource_gathering(trade_history)

            # Production phase
            self._handle_production(trade_history)

            # Trading phase
            self._handle_trading(trade_history)

            # Market adjustment phase
            self._adjust_market()

            # Print round summary
            self._print_round_summary(trade_history)

    def _replenish_resources(self):
        """Natural resource replenishment"""
        for resource in ["wood", "stone"]:
            new_amount = min(
                100,  # Max capacity
                self.market["resource_availability"][resource]
                + self.market["resource_availability"]["replenish_rate"],
            )
            self.market["resource_availability"][resource] = new_amount

    def _handle_resource_gathering(self, trade_history):
        """Carpenters gather resources based on reputation"""
        for carpenter, stats in trade_history.items():
            gather_efficiency = min(1.0, stats["reputation"])
            max_gather = 3 * gather_efficiency  # Better reputation = better gathering

            # Check resource availability
            available_wood = min(
                max_gather, self.market["resource_availability"]["wood"]
            )
            available_stone = min(
                max_gather, self.market["resource_availability"]["stone"]
            )

            if available_wood > 0 and available_stone > 0:
                self.agents[carpenter].inventory["wood"] += available_wood
                self.agents[carpenter].inventory["stone"] += available_stone
                stats["resources_gathered"] += available_wood + available_stone

                # Update environment
                self.market["resource_availability"]["wood"] -= available_wood
                self.market["resource_availability"]["stone"] -= available_stone
                print(
                    f"🌳 {carpenter} gathered {available_wood:.1f} wood, {available_stone:.1f} stone"
                )

    def _handle_production(self, trade_history):
        """Smart production decisions"""
        current_price = self._calculate_market_price()
        production_cost = self._calculate_production_cost()

        for carpenter, stats in trade_history.items():
            # Only produce if profitable and have resources
            if (
                current_price > production_cost * 1.2  # 20% profit margin
                and self.agents[carpenter].inventory["wood"] >= 1
                and self.agents[carpenter].inventory["stone"] >= 1
            ):

                self.agents[carpenter].inventory["wood"] -= 1
                self.agents[carpenter].inventory["stone"] -= 1
                self.agents[carpenter].inventory["hammer"] += 1
                print(
                    f"📦 {carpenter} crafted at price {current_price:.2f} (cost: {production_cost:.2f})"
                )

    def _calculate_market_price(self):
        """Price based on supply/demand and market conditions"""
        base_price = self.market["base_prices"]["hammer"]
        total_hammers = sum(agent.inventory["hammer"] for agent in self.agents.values())

        # Price increases with scarcity
        scarcity_multiplier = 1 + (0.1 * (1 / (total_hammers + 1)))

        # Price affected by resource availability
        resource_pressure = 1 + (
            0.1
            * (
                1
                / (
                    self.market["resource_availability"]["wood"]
                    + self.market["resource_availability"]["stone"]
                    + 1
                )
            )
        )

        return (
            base_price
            * scarcity_multiplier
            * resource_pressure
            * self.market["supply_chain"]["efficiency"]
        )

    def _handle_trading(self, trade_history):
        """Enhanced trading with miner preferences"""
        for miner in ["miner_1", "miner_2"]:
            # Miners consider reputation AND recent success
            carpenter_scores = {
                c: (stats["reputation"] * (1 + stats["trades"] / 10))
                for c, stats in trade_history.items()
            }

            chosen_carpenter = max(carpenter_scores.items(), key=lambda x: x[1])[0]

            if self._execute_trade(chosen_carpenter, miner, trade_history):
                print(
                    f"🤝 {miner} traded with {chosen_carpenter} (rep: {trade_history[chosen_carpenter]['reputation']:.2f})"
                )

    def _adjust_market(self):
        """Handle market conditions and supply chain"""
        if random.random() < self.market["supply_chain"]["disruption_chance"]:
            self.market["supply_chain"]["efficiency"] *= 0.7
            print("🌋 Supply chain disruption! Resource costs increased")
        else:
            # Recovery
            self.market["supply_chain"]["efficiency"] = min(
                1.0,
                self.market["supply_chain"]["efficiency"]
                + self.market["supply_chain"]["recovery_rate"],
            )

    def step(self, action, agent_id, target_id=None, item=None):
        """Execute a single step in the simulation.
        
        Args:
            action (str): Either "craft" or "trade"
            agent_id (str): ID of the acting agent
            target_id (str, optional): ID of the trade target
            item (str, optional): Item being traded
            
        Returns:
            bool: Whether the action was successful
        """
        if action == "craft":
            return self._handle_craft(agent_id)
        elif action == "trade":
            return self._handle_trade(agent_id, target_id, item)
        return False

    def _handle_craft(self, agent_id):
        """Handle crafting action"""
        agent = self.agents[agent_id]
        if agent.can_craft_hammer():
            agent.inventory["wood"] -= 1
            agent.inventory["stone"] -= 1
            agent.inventory["hammer"] += 1
            self.stats["total_crafted"] += 1
            return True
        return False

    def _handle_trade(self, seller_id, buyer_id, item):
        """Handle trading action"""
        seller = self.agents[seller_id]
        buyer = self.agents[buyer_id]
        
        if seller.inventory[item] > 0:
            seller.inventory[item] -= 1
            buyer.inventory[item] += 1
            current_price = self._calculate_market_price()
            self.stats["gdp"] += current_price
            self.stats["total_trades"] += 1
            self.stats["market_prices"].append(current_price)
            return True
        return False

    def simulate_market_cycle(self):
        """Calculate current market price based on supply and demand"""
        return self._calculate_market_price()


# Create our world
world = EnhancedEconomicAdaSociety()


# Scenario 1: Simple Trade
def run_basic_trade():
    print("=== Basic Trade Scenario ===")
    
    # Initialize resources
    world.agents["carpenter_1"].inventory["wood"] = 2
    world.agents["carpenter_1"].inventory["stone"] = 2

    for _ in range(3):
        # Craft and trade cycle
        if world.step("craft", "carpenter_1"):
            print("Carpenter crafted a hammer")
            
        if world.step("trade", "carpenter_1", "miner_1", "hammer"):
            print("Carpenter traded hammer with miner")
            
        price = world.simulate_market_cycle()
        print(f"Current hammer price: {price:.2f}")
        print(f"GDP: {world.stats['gdp']:.2f}")
        print(f"Total trades: {world.stats['total_trades']}")


# Scenario 2: Competition
def run_competition():
    print("\n=== Enhanced Economic Competition Scenario ===")

    # Initialize market tracking
    trade_history = {
        "carpenter_1": {
            "profit": 0,
            "trades": 0,
            "reputation": 1.0,
            "price_modifier": 1.0,
        },
        "carpenter_2": {
            "profit": 0,
            "trades": 0,
            "reputation": 1.0,
            "price_modifier": 1.0,
        },
    }

    supply_chain = {
        "disruption": False,
        "disruption_chance": 0.1,
        "resource_multiplier": 1.0,
    }

    # Initialize carpenters
    for carpenter in ["carpenter_1", "carpenter_2"]:
        world.agents[carpenter].inventory["wood"] = 5
        world.agents[carpenter].inventory["stone"] = 5

    # Run competition rounds
    for round in range(20):
        print(f"\n=== Round {round + 1} ===")

        # Supply chain dynamics
        if random.random() < supply_chain["disruption_chance"]:
            supply_chain["disruption"] = True
            supply_chain["resource_multiplier"] = 1.5
            print("🌋 Supply chain disruption! Resource costs increased")
        else:
            supply_chain["disruption"] = False
            supply_chain["resource_multiplier"] = 1.0

        # Production phase with price competition
        base_price = world.simulate_market_cycle()
        for carpenter in ["carpenter_1", "carpenter_2"]:
            production_cost = 3 * supply_chain["resource_multiplier"]

            # Adjust price based on market position
            if trade_history[carpenter]["trades"] < min(
                t["trades"] for t in trade_history.values()
            ):
                # Undercut competition if falling behind
                trade_history[carpenter]["price_modifier"] = 0.9
            else:
                # Gradually return to normal pricing
                trade_history[carpenter]["price_modifier"] = min(
                    1.0, trade_history[carpenter]["price_modifier"] + 0.05
                )

            effective_price = base_price * trade_history[carpenter]["price_modifier"]

            if effective_price > production_cost:
                if world.step("craft", carpenter):
                    print(
                        f"📦 {carpenter} crafted at price {effective_price:.2f} "
                        + f"(cost: {production_cost:.2f})"
                    )

        # Trading phase with competitive pricing
        for miner in ["miner_1", "miner_2"]:
            carpenter_scores = {
                c: (
                    trade_history[c]["reputation"]
                    * (1 / (base_price * trade_history[c]["price_modifier"]))
                )
                for c in ["carpenter_1", "carpenter_2"]
            }

            chosen_carpenter = max(carpenter_scores.items(), key=lambda x: x[1])[0]

            if world.step("trade", chosen_carpenter, miner, "hammer"):
                effective_price = (
                    base_price * trade_history[chosen_carpenter]["price_modifier"]
                )
                profit = effective_price - production_cost
                trade_history[chosen_carpenter]["profit"] += profit
                trade_history[chosen_carpenter]["trades"] += 1
                trade_history[chosen_carpenter]["reputation"] *= 1.1
                print(
                    f"🤝 {miner} traded with {chosen_carpenter} "
                    + f"(profit: {profit:.2f}, rep: {trade_history[chosen_carpenter]['reputation']:.2f})"
                )
            else:
                trade_history[chosen_carpenter]["reputation"] *= 0.9
                print(f"❌ {chosen_carpenter} failed to deliver to {miner}")

        # Market summary
        print(f"\nMarket Summary:")
        print(f"💰 Base market price: {base_price:.2f}")
        print(
            f"🏭 Supply chain status: {'Disrupted' if supply_chain['disruption'] else 'Normal'}"
        )
        for carpenter, stats in trade_history.items():
            print(
                f"📊 {carpenter}: {stats['trades']} trades, {stats['profit']:.1f} profit, "
                + f"{stats['reputation']:.2f} rep, price mod: {stats['price_modifier']:.2f}"
            )


# Run scenarios
run_basic_trade()
run_competition()
