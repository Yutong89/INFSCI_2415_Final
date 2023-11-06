import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
data = pd.read_csv('diamonds.csv')  # Replace with your file path

# 1. Histograms of different charactors of diamonds
# Columns to plot
columns = ['price', 'carat', 'depth', 'table']

# Iterate over the columns and create individual plots
for i, column in enumerate(columns):
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot histogram without kde
    ax1 = sns.histplot(data[column], kde=False, label=f'{column.capitalize()} Histogram')
    
    # Create a second Y axis for KDE
    ax2 = ax1.twinx()
    
    # Plot kde with a label, red color on the second Y axis
    sns.kdeplot(data[column], label=f'{column.capitalize()} KDE', color='red', ax=ax2)
    
    # Set the title and legends
    plt.title(f"Histogram and KDE for {column.capitalize()}")
    
    # Set legends for histogram and KDE
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.92)) 
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    
    # Save the figure
    plt.savefig(f'hist_kde_{column}.jpg')
    
    # Show the plot
    plt.show()


# 2. heatmap representing the average price for combinations of cut and color
# Define the order for the cuts
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

# Ensure the data['cut'] is a categorical type with the specified order
data['cut'] = pd.Categorical(data['cut'], categories=cut_order, ordered=True)

# Now create the pivot table with the cut order preserved
pivot_table = data.pivot_table(values='price', index='cut', columns='color', aggfunc='mean')

# Plotting the heatmap with the desired cut order
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="YlGnBu")

plt.title('Average Price for Each Cut and Color Combination')
plt.xlabel('Color')
plt.ylabel('Cut')
plt.xticks(rotation=45)  # It makes the color labels more readable
plt.yticks(rotation=0)

# Save the figure
plt.savefig('avg_price_for_cut_and_color.jpg')
plt.show()


# 3. Stem Plot
# Mappings for converting categorical data to numeric
cut_mapping = {'Fair': 20, 'Good': 40, 'Very Good': 60, 'Premium': 80, 'Ideal': 100}
clarity_mapping = {'I1': 10, 'SI2': 20, 'SI1': 30, 'VS2': 40, 'VS1': 50, 'VVS2': 60, 'VVS1': 70, 'IF': 80}

# Apply the mappings to create numeric columns
data['cut_numeric'] = data['cut'].map(cut_mapping)
data['clarity_numeric'] = data['clarity'].map(clarity_mapping)

# Define price bins
price_bins = pd.cut(data['price'], bins=[0, 1000, 2000, 3000, 4000, 5000, 10000, 20000])  # define the interval if needed

# calculate the avg value for each degree
numeric_columns = ['carat', 'depth', 'cut_numeric', 'clarity_numeric']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
grouped = data.groupby(price_bins)[numeric_columns].mean()

# set offset
offsets = {'carat': -0.15, 'cut_numeric': -0.05, 'clarity_numeric': 0.05, 'depth': 0.15}

# set stem plit
fig, ax = plt.subplots(figsize=(12, 8))
scale_factor = 100
for i, column in enumerate(numeric_columns):

    values = grouped[column] if column != 'carat' else grouped[column] * scale_factor
    # add offset
    positions = np.arange(len(grouped)) + offsets[column]
    
    # plot the figure
    label = f'{column.capitalize()} (x{scale_factor})' if column == 'carat' else column.capitalize()
    ax.stem(positions, values, linefmt=f'C{i}-', markerfmt=f'C{i}o', basefmt=" ", label=label)

# set the label and legends
ax.set_xticks(np.arange(len(grouped)))
ax.set_xticklabels([str(interval) for interval in grouped.index.categories])
ax.set_xlabel('Price Range ($)')
ax.set_ylabel('Average Value')
ax.set_title('Stem Plot of Average Carat, Cut, Clarity, and Depth by Price Range')
ax.legend()
# Save the figure
plt.savefig('stem_plot.jpg')
plt.show()

