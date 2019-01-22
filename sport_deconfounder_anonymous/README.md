# Sport deconfounder
A causal inference approach for player estimation.

__Abstract__

We develop a new machine learning method to analyze data about team
sports.  Our goal is to infer the individual contribution of a player
from the collective performance of the team.  We frame the problem as
a causal inference, seeking to answer counterfactual questions of the
form 'What would be the result of a game if player _X_ was substituted
with player _Y_ in the starting lineup'?  To solve the problem, we
develop the _sport deconfounder_, a method that uses causal inference
to evaluate sports players.  The sport deconfounder treats the problem
as a _multiple causal inference_, where each player is a possible
cause that contributes to the outcome of the game.  It analyzes a
dataset of sports results---the outcomes of games between teams and
the player lineup of each team---to infer the causal effect of each
player.  Specifically, it fits both a model of the lineups, a Poisson
tensor factorization, and a model of the outcomes.  Following the
ideas of Wang and Blei (2018), the model of the lineups provide
substitute confounders to aid in the causal inference; these
confounders are then used to debias the model of the outcome.  We
studied the sport deconfounder on data about soccer. It predicts the
score better than adjusted plus-minus (the main existing approach),
and its performance is stable across regular and intervened datasets.
