import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Create a figure and axis without axes and minor grid, and add an arrow
fig, ax = plt.subplots(figsize=(6, 6))


# Draw the 4 unit subsquares
subsquares = [
    patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((1, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor='black'),
    patches.Rectangle((0, 1), 1, 1, linewidth=1, edgecolor='blue', facecolor='black'),
    patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((-1, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((-1, -1), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((-1, 1), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
    patches.Rectangle((1, -1), 1, 1, linewidth=1, edgecolor='blue', facecolor='black'),
    patches.Rectangle((0, -1), 1, 1, linewidth=1, edgecolor='blue', facecolor='none')
]
for subsquare in subsquares:
    ax.add_patch(subsquare)




ax.text(1/2, 3/4, '$P$', horizontalalignment='center', verticalalignment='center', fontsize=22, color='blue')
ax.text(1/2, 1/4, r'$\left(1-\frac{5\delta}{8}\right)·P$', horizontalalignment='center', verticalalignment='center', fontsize=22, color='red')

#ax.text(1/2, 3/2, r'$\delta \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')
ax.text(1/2, -1/2, r'$\frac{\delta}{8} \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')
ax.text(-1/2, 3/2, r'$\frac{\delta}{8} \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')
ax.text(-1/2, -1/2, r'$\frac{\delta}{8} \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')
ax.text(-1/2, 1/2, r'$\frac{\delta}{8} \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')
#ax.text(3/2, 1/2, r'$\delta \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')
#ax.text(3/2, -1/2, r'$\delta \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')
ax.text(3/2, 3/2, r'$\frac{\delta}{8} \cdot P$', horizontalalignment='center', verticalalignment='center', fontsize=25, color='red')






# Set plot limits and properties
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
ax.set_aspect('equal')
#ax.grid(True)
ax.axis('off')  # Turn off the axis
plt.savefig("diffusion.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# Guardar la imagen con buena calidad
