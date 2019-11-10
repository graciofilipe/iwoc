# Prior note:
Below you will find the answers to the posed questions. Before that it's worth nothing that there is a quick exploratory analysis on each file that is logged in the `exploratory_analysis` folder. 
It would have been interesting to plot some of those observations, but for brevity of the exercise I focused elsewhere. 
The analysis shows good things like no missing values, and bad things, like ages up to 200. 

Left to do is also some multivariate exploration (for example are there geographical trends to the sectors, or are the unusually high ages a data quality problem of a certain region). Again I focused on the questions but in a real world exercise exploration would need to be deeper. 

Finally, below are the answers to the questions. All of them come from the code, and the basis for them are the values logged as a result of running the code in the `answers_logs.txt` file. For more detail on how the numbers are derived, feel free to consult the code and logs.  

# Answers


## Which agent made the most calls? [1]
Orange: 2234

## For the leads that received one or more calls, how many calls were received on average? [2]
1.84


## For the leads that signed up, how many calls were received, on average? [2]
2.1


##Which agent had the most signups? Which assumptions did you make? (note that there is a many-to-one relationship between calls and leads) [4]
I gave fractional attribution to each agent for each sign up. That is: for an agent to get a full 1 sign up, they must be the only one calling that lead. If two agents called the person each gets 0.5. If an agent called twice and someone else once, then they get two thirds, etc. 
With that methodology, agent Orange gets the most sign ups with around 308 "sign ups"


## Which agent had the most signups per call? [2]
If I take the number of sign ups (as per methodology above) and divide by the number of calls made, the best agent is Blue, with a success rate of about 23%


## Was the variation between the agents’ signups-per-call statistically significant? Why? [5]
Yes. A chi square contingency table to test for difference in the odds ratios makes the point. What we compare is the ratio of number of sign ups to the number calls across the agents. The resulting p-value is 4e-12.
Of course there are several violations of the assumptions of the test - probably the most significant being the way we count singups per agent (discussed in the previous questions). 
However, given the differences in the results and the very small p-value, the signifance of the difference is likely to hold.


## A lead from which region is most likely to be “interested” in the product? [3]
London, having 5 INTERESTED leads for every 1 NOT INTERESTED lead.



## A lead from which sector is most likely to be “interested” in the product? [1]
Consultancy, having about 2.5 INTERESTED leads for every 1 NOT INTERESTED lead


## Given a lead has already expressed interest and signed up: 

### signups from which region are most likely to be approved? [2]
Scotland, with about half (45%) of sign ups approved. 
### Is this statistically significant? Why? [5]
Yes. A chi square test for the contingency table of approvals to non approvals returns a p-value of about 1e-5.



## Suppose you wanted to pick the 1000 leads most likely to sign up (who have not been called so far), based only on age, sector and region.

### What criteria would you use to pick those leads? [10]
I trained a model to predict the probability of sign up from the call based on sector and region. I excluded age based on the fact that it many untrusthworthy values and based on quick ad-hoc model comparisons. 
Then I can apply the model to the non called leads to calculate the probability of sign up. 
The accuracy of that model is circa 56% with an area under the ROC curve of 0.6. While not a fantastic metric, the practical implications are quantified below.

### In what sense are those an optimal criteria set? [3]
It's the approach that, given past calls and sign up behavior, would have resulted in the most sign ups. 
At a more technical level: it's a selection that minimizes the prediction error on the previously observed behavior. 

### How many signups would you expect to get based on those called leads, assuming they were being called by random agents? [3]
To get this number the process is the following: 
* apply the model to all uncalled leads and get their probabilities of sign up
* find out what is the probability of the 1000th most likely sign up - that is p=0.62
* go to the test dataset and mark as "positive" all the predictions that have a probability greater than 0.69
* calculate the % of true positives among that group: 41%

So following this strategy, over 400 of the leads called are predicted to sign up. That is considerably better than the random strategy which would be expected to have about 290 (the current sign up rate). Obviously, restricting the calls to the top 1000 most likely might represent lower total volume. 

### If you could choose the agents to make those calls, who would you choose? Why? [3]
If one cared only about maximising the sign ups, the sophisticated way to do this would be the following: 
* build a model again which this time includes "agent" as a feature.
* make predictions on the probability of sign up by testing each agent for each lead. As a result we'd know what is the agent that maximises the probability of sign up for each lead
* have each lead be called by the agent that most maximises it's sign up OR
* if one wants to use only one agent, chose the agent that maximises the overall sum of probability over all the leads. 

One important thing to solve in the above process would be "how to assign an agent to a sign up?". Just like before we could use the fractional attribution model, or we could use a last-touch attribution model. 

Alternatively to the more sophisticated approach above, one could chose agent Blue which has the highest success rate. A reason why that might not be optimal, of course, is that the top 1000 calls do *not* have the same profile as a random draw of the previous calls and so there is no guarantee that on these leads Blue is still the best. 

Finally it's worth saying that maximising sign up probability might not be the goal (one might want to divide work, or prioritise speed, etc)