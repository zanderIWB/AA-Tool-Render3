# Import necessary libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import plotly.graph_objects as go

# Define the function to optimize the portfolio
def optimise_sd(n, returns, std_devs, correlations, sd_requirement):

	# Define the expected returns (mu)
	mu = np.array(returns)

	# Define the standard deviations (sigma)
	sigma = np.array(std_devs)

	# Define the correlation matrix (ro)
	if n == 2:
		ro = np.array([[1, correlations[0]],
						[correlations[0],1]])
	if n == 3:
		ro = np.array([[1, correlations[0], correlations[1]],
						[correlations[0], 1, correlations[2]],
						[correlations[1], correlations[2], 1]])

	cov_matrix = np.zeros((n,n))

	for i in range(0,n):
		for j in range(0,n):
			cov_matrix[i,j]=ro[i,j]*sigma[i]*sigma[j]  

	# Define the variables
	weights = cp.Variable(n)

	# Define the objective function
	objective = cp.Maximize(mu @ weights)

	# Define the constraints
	constraints = [weights >= 0,
				   cp.sum(weights) == 1,
				   cp.quad_form(weights, cov_matrix) <= sd_requirement**2]

	# Define the problem
	problem = cp.Problem(objective, constraints)

	# Solve the problem
	problem.solve()

	# Retrieve the optimal weights
	optimal_weights = weights.value

	# Calculate portfolio expected return
	portfolio_expected_return = mu @ optimal_weights

	# Calculate actual portfolio standard deviation
	portfolio_std_dev = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights.T)

	return optimal_weights, portfolio_expected_return, portfolio_std_dev

# Define the function to generate values for the graph
def graphvalues(returns, std_devs, correlations, start_point):
	
	# Define a range of target returns
	target_std_devs = np.linspace(start_point, max(std_devs[0], std_devs[1]), 100)

	# Initialize lists to store portfolio returns and standard deviations
	VC_portfolio_returns = []
	base_portfolio_returns = []

	# Solve for each target return and calculate corresponding portfolio weights
	for sd in target_std_devs:
		VC_weights, VC_expected_return, VC_std_dev = optimise_sd(3,returns, std_devs, correlations,sd)
		base_weights, base_expected_return, base_std_dev = optimise_sd(2,returns[0:2], std_devs[0:2], correlations[0:1],sd)
		VC_portfolio_returns.append(VC_expected_return)
		base_portfolio_returns.append(base_expected_return)
	
	# Convert lists to numpy arrays for plotting
	VC_portfolio_returns = np.array(VC_portfolio_returns)
	base_portfolio_returns = np.array(base_portfolio_returns)
	portfolio_std_devs = np.array(target_std_devs)
	
	return portfolio_std_devs, VC_portfolio_returns, base_portfolio_returns


# Streamlit app starts here
st.title("Portfolio Asset Allocation Tool")

st.header("Introduction")

st.write("This tool has been designed to help visualise how a client's investment portfolio expected returns"
" can increase with the addition of Venture Capital (VC) investment."
" Input the asset allocation of your client's portfolio and see how expected returns can improved.")

st.header("Your Current Portfolio")

st.write("The tool gauges the client's attitude to risk by assessing their current asset allocation and produces a new portfolio including VC with equal risk ."
" Use the slider below to enter your client's equity - bond split in their current portfolio:")


# Collect input values from the user using sliders and number inputs (In this section the default values of mean return, etc can be changed)
equity_percentage = st.slider("Percentage of investment in equity (%)", min_value=0., max_value=100., value=60., step=0.5, format = "%.1f")
bond_percentage = 100 - equity_percentage

st.write(f"Current equity - bond split: {equity_percentage}% in equity and {bond_percentage}% in bond")

with st.expander("Assumptions"):
	st.write("The tool relies on a number of underlying market assumptions. These can be viewed/changed here. Take caution in changing the values from their default.")
	col1, col2, col3 = st.columns(3)
	eq_return = col1.number_input("Equity Expected Return (%)", value=7., min_value=0., step=0.5, format = "%.1f")
	bond_return = col2.number_input("Bond Expected Return (%)", value=4., min_value=0., step=0.5, format = "%.1f")
	col1, col2, col3 = st.columns(3)
	eq_sd = col1.number_input("Equity Standard Deviation (%)", value=15., min_value=0., step=0.5, format = "%.1f")
	bond_sd = col2.number_input("Bond Standard Deviation (%)", value=9., min_value=0., step=0.5, format = "%.1f")
	col1, col2, col3 = st.columns(3)
	eq_bond_cor = col1.number_input("Equity-Bond Correlation", value=0.37, format = "%.2f", min_value=-1., max_value=1.)
	eq_VC_cor = col2.number_input("Equity-VC Correlation", value=0.47, format = "%.2f", min_value=-1., max_value=1.)
	VC_bond_cor = col3.number_input("VC-Bond Correlation", value=0.37, format = "%.2f", min_value=-1., max_value=1.)

with st.expander("Seed and Scale-up Assumptions"):
	st.write("The tool combines early stage, seed and later stage, scale-up venture capital into one asset class."
	" Seed and scale-up investment have different risks and expected returns."
	" Use the slider below to adjust the percentage of seed VC that makes up your client’s proposed venture capital investment.")
	
	seed_percentage = st.slider("Percentage of VC investment in seed (%)", min_value=0., max_value=100., value=20., step=0.5, format = "%.1f")
	scale_percentage = 100 - seed_percentage
	
	col1, col2, col3 = st.columns(3)
	scale_return = col1.number_input("Scale-up Expected Return (%)", value=15., min_value=0., step=0.5, format = "%.1f")
	seed_return = col2.number_input("Seed Expected Return (%)", value=22., min_value=0., step=0.5, format = "%.1f")
	scale_sd = col1.number_input("Scale-up Standard Deviation (%)", value=32., min_value=0., step=0.5, format = "%.1f")
	seed_sd = col2.number_input("Seed Standard Deviation (%)", value=47., min_value=0., step=0.5, format = "%.1f")

	VC_return = (scale_percentage/100)*scale_return+(seed_percentage/100)*seed_return
	VC_sd = (scale_percentage/100)*scale_sd+(seed_percentage/100)*seed_sd

	st.write(f"Combined VC expected return: {round(VC_return,1)}%; Combined VC standard deviation: {round(VC_sd,1)}%")


#Collate the returns, SDs and correlations
returns = [eq_return, bond_return, VC_return]
std_devs = [eq_sd, bond_sd, VC_sd]
correlations = [eq_bond_cor, eq_VC_cor, VC_bond_cor]


# Create the Efficient Frontier plot using Plotly
st.header("Improving the Efficient Frontier")
st.write("The Efficient Frontier displays the optimum expected return of a portfolio at a specific risk level. The impact on the efficient frontier by introducing Venture Capital can be observed below:")

weights = np.array([(equity_percentage/(equity_percentage+bond_percentage)), (bond_percentage/(equity_percentage+bond_percentage))])

cov_matrix = np.array([[eq_sd**2,eq_sd*bond_sd*eq_bond_cor],
						[eq_sd*bond_sd*eq_bond_cor, bond_sd**2]])
												
sd_display = np.sqrt(weights @ cov_matrix @ np.transpose(weights))
return_display = np.array([eq_return, bond_return]) @ weights

display_VC_weights, display_VC_return, display_VC_sd = optimise_sd(3, returns, std_devs, correlations, sd_display)

x, y1, y2 = graphvalues(returns, std_devs, correlations, min(sd_display, std_devs[0], std_devs[1]))

# Create a trace for VC
trace1 = go.Scatter(x=x, y=y1, name="With VC", mode="lines")

# Create a trace for non-VC
trace2 = go.Scatter(x=x, y=y2, name="Without VC", mode="lines")

# Create the figure with both traces
fig = go.Figure(data=[trace1, trace2])

# Add a vertical line

fig.add_shape(
	type="line",
	x0=sd_display,
	y0=min(returns)*1.05,
	x1=sd_display,
	y1=max(y1),
	line=dict(color="green", width=2, dash = "dash")
)

fig.add_annotation(x=sd_display, y=min(returns), text="Current Portfolio Risk", showarrow=False)

# Update layout with axis labels and title
fig.update_layout(xaxis_title="Portfolio Standard Deviation (%)", yaxis_title="Portfolio Expected Return (%)")

# Format hover label to display 3 decimal places
fig.update_traces(hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}')

# Display the figure
st.plotly_chart(fig)

# Display the improvement in expected return with VC
st.write(f"Your current portfolio has annual expected returns of {round(return_display,1)}%. By introducing Venture Capital into your portfolio, "
f"your expected return could increase to {round(display_VC_return,1)}%, an increase of "
f"{round(display_VC_return-return_display,1)}%. There is no increase in the risk of the portfolio (portfolio standard deviation "
f"remains {round(display_VC_sd,1)}%).")


st.header("Asset Allocation")

st.write("To achieve this increase in expected return, the allocation of your client's assets should be altered. Your client's current asset allocation and the revised allocation including VC are shown below. ")
st.write("Current asset allocation:")

fig, ax = plt.subplots(1, figsize=(6, 1.5))

# Display the current asset allocation using Matplotlib
ax.barh(y=0, width=equity_percentage, color='blue', label='Equity')
ax.barh(y=0, width=bond_percentage, left=equity_percentage, color='orange', label='Bonds')

ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([0])
ax.set_yticklabels([' '])
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

ax.text(equity_percentage / 2, 0, "{:.0f}%".format(equity_percentage), ha='center', va='center', color='white', fontsize=12)
ax.text(equity_percentage + bond_percentage / 2, 0, "{:.0f}%".format(bond_percentage), ha='center', va='center', color='white', fontsize=12)

ax.text(equity_percentage / 2, 0.2, 'Equity', ha='center', va='bottom', color='white')
ax.text(equity_percentage + bond_percentage / 2, 0.2, 'Bonds', ha='center', va='bottom', color='white')

st.pyplot(fig)

# Display the new asset allocation with VC using Matplotlib
st.write("New asset allocation with venture capital (VC):")

fig, ax = plt.subplots(1, figsize=(6, 1.5))

VC_weights = {'Equity': round(display_VC_weights[0]*100,1), 'Bonds': round(display_VC_weights[1]*100,1), 'VC': round(display_VC_weights[2]*100,1)}
colors = ['blue', 'orange', 'green']

ax.barh(y=0, width=100, color='#d3d3d3')

x_pos = 0
for asset, percentage in VC_weights.items():
	width = percentage
	ax.barh(y=0, width=width, left=x_pos, color=colors.pop(0))
	ax.text(x_pos + width / 2, 0, "{:.0f}%".format(percentage), ha='center', va='center', color='white', fontsize=12)
	ax.text(x_pos + width / 2, 0.2, asset, ha='center', va='bottom', color='white')
	x_pos += width

ax.set_xlim(0, 100)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([0])
ax.set_yticklabels([' '])
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

st.pyplot(fig)

#Disclaimers Expander
with st.expander("Disclaimers"):
	st.write(" ")


