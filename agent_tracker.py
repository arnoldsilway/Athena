# agent_tracker.py - Track Agent Actions and Rewards

import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Action:
    """Represents an agent action"""
    name: str
    params: Dict
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


@dataclass
class Reward:
    """Represents a reward signal"""
    value: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self):
        sign = "+" if self.value >= 0 else ""
        return f"{sign}{self.value:.2f} - {self.reason}"


class AgentTracker:
    """
    Tracks agent actions and rewards for reinforcement learning demonstration
    
    Actions:
    - search_papers: Search for research papers
    - extract_text: Extract text from PDF
    - build_index: Build semantic index
    - answer_query: Answer user question
    - compare_docs: Compare documents
    
    Rewards:
    - Task completion: +10
    - Fast response: +5
    - High quality: +8
    - Error occurred: -5
    - Timeout: -3
    """
    
    def __init__(self):
        self.actions: List[Action] = []
        self.rewards: List[Reward] = []
        self.episode_start = time.time()
        self.total_reward = 0.0
        
    def log_action(self, action_name: str, **params) -> Action:
        """Log an agent action"""
        action = Action(name=action_name, params=params)
        self.actions.append(action)
        print(f"🤖 ACTION: {action}")
        return action
    
    def add_reward(self, value: float, reason: str) -> Reward:
        """Add a reward signal"""
        reward = Reward(value=value, reason=reason)
        self.rewards.append(reward)
        self.total_reward += value
        
        sign = "🟢" if value >= 0 else "🔴"
        print(f"{sign} REWARD: {reward}")
        
        return reward
    
    def get_episode_summary(self) -> Dict:
        """Get summary of current episode"""
        duration = time.time() - self.episode_start
        
        return {
            'total_actions': len(self.actions),
            'total_rewards': len(self.rewards),
            'cumulative_reward': self.total_reward,
            'episode_duration': duration,
            'average_reward': self.total_reward / max(len(self.rewards), 1),
            'actions_per_minute': (len(self.actions) / duration) * 60,
        }
    
    def get_recent_history(self, n: int = 5) -> List[tuple]:
        """Get recent action-reward pairs"""
        history = []
        
        # Pair actions with rewards (simplified - assumes 1:1 mapping)
        for i in range(min(n, len(self.actions))):
            action = self.actions[-(i+1)]
            reward = self.rewards[-(i+1)] if i < len(self.rewards) else None
            history.append((action, reward))
        
        return history
    
    def display_state(self):
        """Display current agent state"""
        print("\n" + "="*70)
        print(" AGENT STATE")
        print("="*70)
        
        summary = self.get_episode_summary()
        
        print(f"\n Episode Statistics:")
        print(f"   Total Actions: {summary['total_actions']}")
        print(f"   Total Rewards: {summary['total_rewards']}")
        print(f"   Cumulative Reward: {summary['cumulative_reward']:.2f}")
        print(f"   Average Reward: {summary['average_reward']:.2f}")
        print(f"   Episode Duration: {summary['episode_duration']:.1f}s")
        
        print(f"\n Recent History (last 5):")
        history = self.get_recent_history(5)
        
        for i, (action, reward) in enumerate(history, 1):
            print(f"\n   {i}. Action: {action.name}")
            if action.params:
                print(f"      Params: {action.params}")
            if reward:
                print(f"      Reward: {reward}")
        
        print("\n" + "="*70)
    
    def export_trajectory(self, filename: str = "agent_trajectory.json"):
        """Export complete trajectory for analysis"""
        trajectory = {
            'episode_start': self.episode_start,
            'episode_duration': time.time() - self.episode_start,
            'total_reward': self.total_reward,
            'actions': [
                {
                    'name': a.name,
                    'params': a.params,
                    'timestamp': a.timestamp
                }
                for a in self.actions
            ],
            'rewards': [
                {
                    'value': r.value,
                    'reason': r.reason,
                    'timestamp': r.timestamp
                }
                for r in self.rewards
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(trajectory, f, indent=2)
        
        print(f"✅ Trajectory saved to {filename}")
    
    def reset_episode(self):
        """Start a new episode"""
        print("\n🔄 Starting new episode...")
        self.actions.clear()
        self.rewards.clear()
        self.episode_start = time.time()
        self.total_reward = 0.0


# Reward calculation functions
class RewardCalculator:
    """Calculate rewards based on task performance"""
    
    @staticmethod
    def task_completion(success: bool) -> float:
        """Reward for completing a task"""
        return 10.0 if success else -5.0
    
    @staticmethod
    def response_time(duration: float, threshold: float = 5.0) -> float:
        """Reward based on response time"""
        if duration < threshold:
            return 5.0
        elif duration < threshold * 2:
            return 2.0
        else:
            return -3.0
    
    @staticmethod
    def quality_score(similarity: float) -> float:
        """Reward based on quality (e.g., similarity score)"""
        if similarity > 0.8:
            return 8.0
        elif similarity > 0.6:
            return 5.0
        elif similarity > 0.4:
            return 2.0
        else:
            return -2.0
    
    @staticmethod
    def error_penalty() -> float:
        """Penalty for errors"""
        return -5.0
    
    @staticmethod
    def user_feedback(rating: int) -> float:
        """Reward from user feedback (1-5 stars)"""
        return (rating - 3) * 3.0  # -6 to +6


# =====================================================================
#  DEMONSTRATION
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" AGENT REWARD & ACTION TRACKING DEMO")
    print("=" * 70)
    
    # Initialize tracker
    tracker = AgentTracker()
    calc = RewardCalculator()
    
    print("\n Simulating research assistant workflow...\n")
    
    # Episode 1: Successful paper search
    print("--- Task 1: Search for papers ---")
    start = time.time()
    
    tracker.log_action("search_papers", query="transformer attention", max_results=5)
    time.sleep(0.5)  # Simulate work
    
    duration = time.time() - start
    tracker.add_reward(calc.task_completion(True), "Papers found successfully")
    tracker.add_reward(calc.response_time(duration, 2.0), f"Response time: {duration:.2f}s")
    
    # Episode 2: Extract and index PDF
    print("\n--- Task 2: Process document ---")
    start = time.time()
    
    tracker.log_action("extract_text", filename="paper.pdf", pages=10)
    time.sleep(0.3)
    
    tracker.log_action("build_index", chunk_size=500, chunks=50)
    time.sleep(0.4)
    
    duration = time.time() - start
    tracker.add_reward(calc.task_completion(True), "Document processed")
    tracker.add_reward(calc.response_time(duration, 1.0), f"Processing time: {duration:.2f}s")
    
    # Episode 3: Answer query
    print("\n--- Task 3: Answer question ---")
    start = time.time()
    
    tracker.log_action("answer_query", 
                      query="What is the main contribution?",
                      context_chunks=3)
    time.sleep(0.6)
    
    duration = time.time() - start
    similarity = 0.85  # High quality answer
    
    tracker.add_reward(calc.task_completion(True), "Answer generated")
    tracker.add_reward(calc.quality_score(similarity), f"Quality: {similarity:.0%}")
    tracker.add_reward(calc.user_feedback(5), "User gave 5 stars")
    
    # Episode 4: Error case
    print("\n--- Task 4: Failed operation ---")
    
    tracker.log_action("compare_docs", doc1="paper1.pdf", doc2="paper2.pdf")
    time.sleep(0.2)
    
    tracker.add_reward(calc.task_completion(False), "Comparison failed")
    tracker.add_reward(calc.error_penalty(), "Document not found error")
    
    # Display final state
    tracker.display_state()
    
    # Export trajectory
    tracker.export_trajectory("demo_trajectory.json")
    
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    
    print("\n Key Concepts Demonstrated:")
    print("    Actions: Logged with parameters")
    print("    Rewards: Calculated based on performance")
    print("    State: Tracked throughout episode")
    print("    Metrics: Cumulative reward, average, etc.")
    print("    Export: Save trajectory for analysis")
    
    print("\n🔗 Integration with Athena:")
    print("   1. Import AgentTracker in main.py")
    print("   2. Log actions before each operation")
    print("   3. Calculate rewards after completion")
    print("   4. Display state in Streamlit UI")
    print("   5. Use for debugging and optimization")