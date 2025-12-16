import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure with 8 subplots (2 rows x 4 columns)
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

# Define the 8 different combinations of face colors for the subsquares (excluding the lower-left)
combinations = [
    ('none', 'none', 'none'),
    ('none', 'none', 'black'),
    ('none', 'black', 'none'),
    ('none', 'black', 'black'),
    ('black', 'none', 'none'),
    ('black', 'none', 'black'),
    ('black', 'black', 'none'),
    ('black', 'black', 'black')
]

# Iterate over each subplot and combination
for ax, (color1, color2, color3) in zip(axes.flatten(), combinations):
    # Main square
    main_square = patches.Rectangle((0, 0), 2, 2, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(main_square)
    
    # 4 subsquares
    # Lower-left always 'blue'
    subsquares = [
        patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor='none'),
        patches.Rectangle((1, 0), 1, 1, linewidth=1, edgecolor='blue', facecolor=color1),
        patches.Rectangle((0, 1), 1, 1, linewidth=1, edgecolor='blue', facecolor=color2),
        patches.Rectangle((1, 1), 1, 1, linewidth=1, edgecolor='blue', facecolor=color3)
    ]
    for subsquare in subsquares:
        ax.add_patch(subsquare)

    ax.text(1/2, 3/4, '$P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='blue')

    if (color1,color2,color3)==('black','black','black'):
        ax.text(1/2, 1/4, '$P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')
    elif (color1,color2,color3)==('none','none','none'):
        ax.text(1/2, 1/4, '$(1-WN)$\n$·(1-WE)·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')
    elif (color1,color2)==('black','black'):
        ax.text(1/2, 1/4, '$(1-WN·WE)·P$', horizontalalignment='center', verticalalignment='center', fontsize=17, color='red')
    elif color1=='black':
        ax.text(1/2, 1/4, '$(1-WN)·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')
    elif color2=='black':
        ax.text(1/2, 1/4, '$(1-WE)·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')
    else:
        ax.text(1/2, 1/4, '$(1-WN-WE$\n$+2WN·WE)·P$', horizontalalignment='center', verticalalignment='center', fontsize=18, color='red')
    


    if color1=='none':
        if (color2,color3)==('black','black'):
            ax.text(3/2, 1/2, '$WE·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')
        else:
            ax.text(3/2, 1/2, '$(1-WN)$\n$·WE·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')
            
    if color2=='none':
        if (color1,color3)==('black','black'):
            ax.text(1/2, 3/2, '$WN·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')
        else:
            ax.text(1/2, 3/2, '$(1-WE)$\n$·WN·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')

    if color3=='none':
        ax.text(3/2, 3/2, '$WN·WE·P$', horizontalalignment='center', verticalalignment='center', fontsize=23, color='red')



    # Set plot properties for each subplot
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off the axis

# Display the plot
plt.tight_layout()
plt.savefig("wind2.png", dpi=300, bbox_inches='tight')
plt.show()
