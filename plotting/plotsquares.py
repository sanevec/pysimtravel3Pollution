import matplotlib.pyplot as plt
import matplotlib.patches as patches

alpha = 0.4
beta = 0.2

# Create a figure and axis without axes and minor grid, and add an arrow
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the main square
main_square = patches.Rectangle((0, 0), 2, 2, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(main_square)

# Draw the 4 unit subsquares
subsquares = [
    patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((1, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((0, 1), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='blue', facecolor='none')
]
for subsquare in subsquares:
    ax.add_patch(subsquare)

# Draw the moving square with the lower left corner at (alpha, beta)
moving_square = [
    patches.Rectangle((alpha, beta), 1, 1, linewidth=1, edgecolor='red', facecolor='none'),
    patches.Rectangle((alpha+1, beta), 1, 1, linewidth=1, edgecolor='red', facecolor='none'),
    patches.Rectangle((alpha+1, beta+1), 1, 1, linewidth=1, edgecolor='red', facecolor='none'),
    patches.Rectangle((alpha, beta+1), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
]
for subsquare in moving_square:
    ax.add_patch(subsquare)

# Add an arrow pointing to the lower left square
ax.annotate('$P$', xy=(1/2,1/2), xytext=(-0.5, 0.5), 
            arrowprops=dict(facecolor='blue', shrink=0, connectionstyle="arc3,rad=0"),#0.05
            horizontalalignment='center', verticalalignment='center', fontsize=20)
ax.annotate('$WN·WE·P$', xy=(1+alpha/2,1+beta/2), xytext=(2.5, 2.5),#xytext=((3+alpha)/2, (3+beta)/2), 
            arrowprops=dict(facecolor='red', shrink=0, connectionstyle="arc3,rad=0"),
            horizontalalignment='center', verticalalignment='center', fontsize=20)
ax.annotate('$(1-WN)·WE·P$', xy=(1+alpha/2,(1+beta)/2), xytext=(2.5,-0.5),#xytext=((3+alpha)/2, beta/2), 
            arrowprops=dict(facecolor='red', shrink=0, connectionstyle="arc3,rad=0"),
            horizontalalignment='center', verticalalignment='center', fontsize=20)
ax.annotate('$WN·(1-WE)·P$', xy=((1+alpha)/2,1+beta/2), xytext=(-0.5,2.5),#xytext=(alpha/2, (3+beta)/2), 
            arrowprops=dict(facecolor='red', shrink=0, connectionstyle="arc3,rad=0"),
            horizontalalignment='center', verticalalignment='center', fontsize=20)
ax.annotate('$(1-WN)·(1-WE)·P$', xy=((1+alpha)/2,(1+beta)/2), xytext=(-0.5,-0.5),#xytext=(alpha/2, beta/2), 
            arrowprops=dict(facecolor='red', shrink=0, connectionstyle="arc3,rad=0"),
            horizontalalignment='center', verticalalignment='center', fontsize=20)


# Set plot limits and properties
ax.set_xlim(-1, 3)
ax.set_ylim(-1,3)
ax.set_aspect('equal')
#ax.grid(True)
ax.axis('off')  # Turn off the axis
plt.savefig("wind.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
