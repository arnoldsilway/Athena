# agent_ui.py - Streamlit UI for Agent Tracking

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agent_tracker import AgentTracker, RewardCalculator
import pandas as pd
from datetime import datetime


def render_agent_dashboard(tracker: AgentTracker):
    """Render agent tracking dashboard in Streamlit"""
    
    st.markdown("## 🤖 Agent Performance Dashboard")
    st.markdown("*Real-time monitoring of agent actions and rewards*")
    
    # Summary metrics
    summary = tracker.get_episode_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Actions", summary['total_actions'])
    
    with col2:
        reward_delta = summary['cumulative_reward']
        st.metric("Cumulative Reward", f"{reward_delta:.1f}", 
                 delta=f"{reward_delta:+.1f}")
    
    with col3:
        st.metric("Average Reward", f"{summary['average_reward']:.2f}")
    
    with col4:
        st.metric("Episode Duration", f"{summary['episode_duration']:.1f}s")
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Reward Timeline", 
        " Action History", 
        " Performance Metrics",
        " Detailed View"
    ])
    
    # TAB 1: Reward Timeline
    with tab1:
        st.markdown("### Cumulative Reward Over Time")
        
        if tracker.rewards:
            # Calculate cumulative rewards
            cumulative = []
            total = 0
            timestamps = []
            
            for reward in tracker.rewards:
                total += reward.value
                cumulative.append(total)
                timestamps.append(reward.timestamp - tracker.episode_start)
            
            # Create figure
            fig = go.Figure()
            
            # Add cumulative line
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=cumulative,
                mode='lines+markers',
                name='Cumulative Reward',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                xaxis_title="Time (seconds)",
                yaxis_title="Cumulative Reward",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Reward distribution
            st.markdown("### Reward Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                positive = sum(1 for r in tracker.rewards if r.value > 0)
                negative = sum(1 for r in tracker.rewards if r.value < 0)
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Positive', 'Negative'],
                    values=[positive, negative],
                    marker_colors=['#06D6A0', '#EF476F']
                )])
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Individual rewards
                reward_values = [r.value for r in tracker.rewards]
                
                fig = go.Figure(data=[go.Bar(
                    x=list(range(len(reward_values))),
                    y=reward_values,
                    marker_color=['#06D6A0' if v > 0 else '#EF476F' for v in reward_values]
                )])
                
                fig.update_layout(
                    xaxis_title="Reward Index",
                    yaxis_title="Reward Value",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rewards recorded yet")
    
    # TAB 2: Action History
    with tab2:
        st.markdown("### Recent Actions")
        
        if tracker.actions:
            history = tracker.get_recent_history(10)
            
            for i, (action, reward) in enumerate(history, 1):
                with st.expander(
                    f"Action {len(tracker.actions) - i + 1}: {action.name}", 
                    expanded=(i <= 3)
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Action:** `{action.name}`")
                        
                        if action.params:
                            st.markdown("**Parameters:**")
                            for key, value in action.params.items():
                                st.write(f"  • {key}: `{value}`")
                        
                        timestamp = datetime.fromtimestamp(action.timestamp)
                        st.caption(f"Time: {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
                    
                    with col2:
                        if reward:
                            color = "green" if reward.value >= 0 else "red"
                            st.markdown(f"**Reward:** :{color}[{reward.value:+.1f}]")
                            st.caption(reward.reason)
                        else:
                            st.caption("No reward logged")
        else:
            st.info("No actions recorded yet")
    
    # TAB 3: Performance Metrics
    with tab3:
        st.markdown("### Performance Analysis")
        
        if tracker.actions:
            # Action frequency
            action_counts = {}
            for action in tracker.actions:
                action_counts[action.name] = action_counts.get(action.name, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Action Distribution")
                
                df = pd.DataFrame({
                    'Action': list(action_counts.keys()),
                    'Count': list(action_counts.values())
                })
                
                fig = go.Figure(data=[go.Bar(
                    x=df['Action'],
                    y=df['Count'],
                    marker_color='#118AB2'
                )])
                
                fig.update_layout(
                    xaxis_title="Action Type",
                    yaxis_title="Count",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Reward Statistics")
                
                if tracker.rewards:
                    reward_stats = {
                        'Total': len(tracker.rewards),
                        'Positive': sum(1 for r in tracker.rewards if r.value > 0),
                        'Negative': sum(1 for r in tracker.rewards if r.value < 0),
                        'Max': max(r.value for r in tracker.rewards),
                        'Min': min(r.value for r in tracker.rewards),
                        'Mean': sum(r.value for r in tracker.rewards) / len(tracker.rewards)
                    }
                    
                    for key, value in reward_stats.items():
                        if key in ['Max', 'Min', 'Mean']:
                            st.metric(key, f"{value:.2f}")
                        else:
                            st.metric(key, value)
                else:
                    st.info("No rewards yet")
            
            # Success rate
            if tracker.rewards:
                st.markdown("#### Success Metrics")
                
                success_rewards = [r for r in tracker.rewards if "success" in r.reason.lower() or r.value > 5]
                success_rate = len(success_rewards) / len(tracker.rewards) * 100
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=success_rate,
                    title={'text': "Success Rate"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#06D6A0"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FFE5E5"},
                            {'range': [50, 75], 'color': "#FFF8DC"},
                            {'range': [75, 100], 'color': "#E8F5E9"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data to analyze yet")
    
    # TAB 4: Detailed View
    with tab4:
        st.markdown("### 🔍 Complete Episode Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Actions Log")
            
            if tracker.actions:
                actions_data = []
                for i, action in enumerate(tracker.actions, 1):
                    actions_data.append({
                        '#': i,
                        'Action': action.name,
                        'Time': f"{action.timestamp - tracker.episode_start:.2f}s"
                    })
                
                df = pd.DataFrame(actions_data)
                st.dataframe(df, use_container_width=True, height=300)
            else:
                st.info("No actions yet")
        
        with col2:
            st.markdown("#### Rewards Log")
            
            if tracker.rewards:
                rewards_data = []
                for i, reward in enumerate(tracker.rewards, 1):
                    rewards_data.append({
                        '#': i,
                        'Value': f"{reward.value:+.1f}",
                        'Reason': reward.reason[:30] + '...' if len(reward.reason) > 30 else reward.reason
                    })
                
                df = pd.DataFrame(rewards_data)
                st.dataframe(df, use_container_width=True, height=300)
            else:
                st.info("No rewards yet")
        
        # Export options
        st.markdown("---")
        st.markdown("#### 💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Trajectory"):
                filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                tracker.export_trajectory(filename)
                st.success(f"Saved to {filename}")
        
        with col2:
            if st.button("🔄 Reset Episode"):
                tracker.reset_episode()
                st.success("Episode reset!")
                st.rerun()


# =====================================================================
# 🧪 DEMO APPLICATION
# =====================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Agent Tracker Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Agent Tracking System Demo")
    
    # Initialize tracker in session state
    if 'agent_tracker' not in st.session_state:
        st.session_state.agent_tracker = AgentTracker()
        st.session_state.calc = RewardCalculator()
    
    tracker = st.session_state.agent_tracker
    calc = st.session_state.calc
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("##  Simulate Actions")
        
        action_type = st.selectbox(
            "Action Type",
            ["search_papers", "extract_text", "build_index", 
             "answer_query", "compare_docs"]
        )
        
        if st.button("▶️ Execute Action", type="primary"):
            # Simulate action
            if action_type == "search_papers":
                tracker.log_action(action_type, query="AI research", max_results=5)
                tracker.add_reward(calc.task_completion(True), "Papers retrieved")
                tracker.add_reward(calc.response_time(1.2), "Fast response")
            
            elif action_type == "extract_text":
                tracker.log_action(action_type, filename="paper.pdf", pages=10)
                tracker.add_reward(calc.task_completion(True), "Text extracted")
                tracker.add_reward(calc.quality_score(0.9), "High quality extraction")
            
            elif action_type == "build_index":
                tracker.log_action(action_type, chunks=50, dimension=384)
                tracker.add_reward(calc.task_completion(True), "Index built")
            
            elif action_type == "answer_query":
                tracker.log_action(action_type, query="What is the main finding?")
                tracker.add_reward(calc.task_completion(True), "Answer generated")
                tracker.add_reward(calc.user_feedback(4), "User rated 4 stars")
            
            else:
                tracker.log_action(action_type, doc1="A", doc2="B")
                tracker.add_reward(calc.task_completion(False), "Comparison failed")
                tracker.add_reward(calc.error_penalty(), "Document not found")
            
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🎲 Random Action"):
            import random
            
            actions = ["search_papers", "extract_text", "build_index", "answer_query"]
            action = random.choice(actions)
            
            tracker.log_action(action, param=random.randint(1, 10))
            
            reward_val = random.uniform(-5, 10)
            reason = random.choice([
                "Task completed",
                "Fast response", 
                "Good quality",
                "Error occurred"
            ])
            
            tracker.add_reward(reward_val, reason)
            st.rerun()
    
    # Main dashboard
    render_agent_dashboard(tracker)
    
    # Footer
    st.markdown("---")
    st.caption("🤖 Agent Tracker v1.0 | Built for RL Demonstrations")