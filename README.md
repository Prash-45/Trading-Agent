Chapter 1 - The Wrong Question 

Most trading projects begin like this:

Historical Data --> Predict Tomorrow's Price --> Buy / Sell

But after working on multiple prediction models, I realized something was missing.

Even if a model predicts tomorrow's price correctly...

1) How much should I invest?
2) Should I hold my position?
3) Should I sell now or wait?
4) How does today's decision affect tomorrow?

Prediction wasn't solving the complete problem 

<img width="1535" height="1024" alt="WhatsApp Image 2026-07-28 at 21 28 32" src="https://github.com/user-attachments/assets/56c9d83b-b645-482a-8cd9-697983192ef7" />


Chapter 2 — Thought of teaching an Agent Instead of Building a Predictor

The idea evolved into this:

Historical Market Data --> Feature Engineering --> Custom Trading Environment --> PPO Agent ---> Continuous Actions --> Portfolio Reward


Chapter 3 — Building the Environment

This was the longest part of the project.

The neural network wasn't my first concern. The environment was.

I had to define:

State --> Action --> Environment Dynamics --> Reward

<img width="1024" height="1536" alt="WhatsApp Image 2026-07-28 at 22 14 44" src="https://github.com/user-attachments/assets/27fb1d71-1f23-4033-86e7-bf48e1bdc657" />



Chapter 4 — Training Day

Finally...

The environment passed Gymnasium validation.

The PPO agent started training.

Thousands of interactions later...

I had my first trained trading agent.

For the first time, I wasn't predicting prices.

I was training a policy.

<img width="1170" height="380" alt="image" src="https://github.com/user-attachments/assets/2a9c9c52-21cc-40bc-9067-3d85ea9b4ec4" />


Chapter 5 — Reality Hits

Evaluation day.

I expected to see something like this.

Buy --> Hold --> Sell --> Buy --> Hold

Instead...

The policy produced --> +1,+1,+1,+1,+1,+1 (selling always)

<img width="1170" height="232" alt="WhatsApp Image 2026-07-28 at 22 44 43" src="https://github.com/user-attachments/assets/763650f2-20a7-4dc0-9555-af31ac995056" />

Chapter 6 — The Investigation

At first I assumed there was a bug.

So I investigated everything.

Environment
Observation Space
Action Space
check_env()
Training
Evaluation

Nothing was broken.

The code worked.

The policy was doing exactly what it had learned.

<img width="1168" height="460" alt="image" src="https://github.com/user-attachments/assets/683f8995-dea3-47ca-9fa1-e3d2621b362a" />


Chapter 7 — The Real Problem

The biggest lesson wasn't about PPO.

It was about problem formulation.

I realized that an RL agent doesn't learn the strategy I imagine.

It learns the strategy my reward function encourages.

My reward encouraged maximizing portfolio growth.

My training data mostly contained upward-trending stocks.

The easiest solution became:

Always Buy

The model didn't fail.

It optimized exactly what I asked it to optimize.

This was the biggest lesson of the entire project.

<img width="1166" height="920" alt="image" src="https://github.com/user-attachments/assets/c4c0e969-7c71-4c6f-91b5-801b728162f4" />


Chapter 8 — Results

Portfolio Value

<img width="1164" height="888" alt="WhatsApp Image 2026-07-28 at 22 54 21" src="https://github.com/user-attachments/assets/3952d4cd-2426-4bc0-b4aa-878f0d9d83b7" />

     ↓

Buy & Hold Comparison

<img width="1166" height="478" alt="WhatsApp Image 2026-07-28 at 22 55 43" src="https://github.com/user-attachments/assets/42399fa1-3901-4599-807d-3ad3b55246f1" />

     ↓

Action Distribution

<img width="1096" height="826" alt="image" src="https://github.com/user-attachments/assets/9c5f3bb7-eb38-4278-a706-404721c21c4b" />


 
Important Note -->

One of the biggest lessons from this project reminded me of Abraham Wald's WWII aircraft story. Engineers wanted to reinforce the parts of returning planes covered in bullet holes. Wald pointed out they should reinforce the areas without bullet holes, because planes hit there never returned.

The agent always buying wasn't necessarily evidence that PPO had failed. It was evidence that the reward function, market regime, and environment design allowed that behavior to become the easiest way to maximize reward.

I made a similar mistake. I initially blamed the PPO agent for always buying. After debugging, I realized the agent wasn't wrong—it was simply optimizing the reward function and environment I had designed.
The lesson: In AI, don't just analyze the model's behavior. Analyze the objective and environment that produced it. A model learns exactly what you reward, not necessarily what you intend.
