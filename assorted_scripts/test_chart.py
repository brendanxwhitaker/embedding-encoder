# imports

import io
import os
import sys
import csv
import math

import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.font_manager as fm
import matplotlib.transforms as transforms

import pandas as pd
import numpy as np

#DATA PREPROCESSING

rows = []
with open("shuffled_6ary_data.csv", 'r') as csvfile:
    rows = csv.reader(csvfile, delimiter=',', quotechar='|')
    rows = list(rows)
numerical_rows = rows[1:len(rows)]
scores_array = np.array(numerical_rows)
scores_array = np.vstack(scores_array)
print(scores_array)

# delete last column, it's empty
scores_array = np.delete(scores_array, 4, 1)
scores = pd.DataFrame(scores_array)
scores.columns = ['shuffle_ratio',
                  'Frequency drop score',
                  'Silhouette score',
                  'Naive tree distance score']

scores = scores.astype(float)

# TEMP TRANSFORM
scores.loc[:,'Silhouette score'] = (scores.loc[:,'Silhouette score'] + 1) / 2

print(scores)


# PLOTTING

#================================================
# PARAMETERS
#================================================

# dataframe
df = scores

# x-axis
xaxis = 'shuffle_ratio'

# y-axis
yaxis = None

# text
title_text = "Cleanliness scores"
subtitle_text = "6-ary tree with height 2."
xlabel = "Shuffle ratio"
ylabel = "Score"
banner_text = "©Computation Institute"

# edges of plot in figure (padding)
top = 0.8
bottom = 0.18
left = 0.1
right = 0.95

# change title_pad to adjust xpos of title in pixels
# + is left, - is right
title_pad = 0 

# opacity
text_opacity = 0.75
xaxis_opacity = 0.7

# sizing
tick_label_size = 7
legend_size = 8
axis_label_size = 10
banner_text_size = 10

# import font
prop = fm.FontProperties(fname='DecimaMonoPro.ttf')
prop2 = fm.FontProperties(fname='apercu_medium_pro.otf')
prop3 = fm.FontProperties(fname='Apercu.ttf')
prop4 = fm.FontProperties(fname='Apercu.ttf', size=legend_size)

#================================================
# END OF PARAMETERS
#================================================

# figure initialization
fig, ax = plt.subplots(figsize=(6, 6))
plt.sca(ax)
style.use('fivethirtyeight')

#===PLOT===
graph = df.plot(x=xaxis, 
                y=yaxis,
                ax=ax, 
                use_index=True, 
                legend=True)

# add axis labels
plt.xlabel(xlabel, 
           fontproperties=prop3, 
           fontsize = axis_label_size, 
           alpha=text_opacity)
plt.ylabel(ylabel, 
           fontproperties=prop3, 
           fontsize = axis_label_size, 
           alpha=text_opacity)

# change font of legend
L = graph.legend(prop={'size': legend_size})
plt.setp(L.texts, fontproperties=prop4, alpha=text_opacity)

# set size of tick labels
graph.tick_params(axis = 'both', 
                  which = 'major', 
                  labelsize = tick_label_size)

# set fontname for tick labels
for tick in graph.get_xticklabels():
    tick.set_fontname("DecimaMonoPro")
for tick in graph.get_yticklabels():
    tick.set_fontname("DecimaMonoPro")
    
graph.set_yticklabels(["apple", "mango", "orange","blueberry","avocado","tomato","cherry","grapefruit", "mango", "orange","blueberry","avocado","tomato","cherry","grapefruit", "mango", "orange","blueberry","avocado","tomato","cherry","grapefruit", "mango", "orange","blueberry","avocado","tomato","cherry","grapefruit", "mango", "orange","blueberry","avocado","tomato","cherry","grapefruit"])
plt.yticks(np.arange(0, 40, 1.0))
    
# set color for tick labels
[t.set_color('#303030') for t in ax.xaxis.get_ticklabels()]
[t.set_color('#303030') for t in ax.yaxis.get_ticklabels()]

# create bolded x-axis
graph.axhline(y = 0, 
              color = 'black', 
              linewidth = 1.3, 
              alpha = xaxis_opacity)

# transforms the x axis to figure fractions, and leaves y axis in pixels
xfig_trans = transforms.blended_transform_factory(fig.transFigure, transforms.IdentityTransform())
yfig_trans = transforms.blended_transform_factory(transforms.IdentityTransform(), fig.transFigure)

# banner positioning
banner_y = math.ceil(banner_text_size * 0.6)

# banner text
banner = plt.annotate(banner_text, 
         xy=(0.02, banner_y), 
         xycoords=xfig_trans,
         fontsize = banner_text_size, 
         color = '#FFFFFF', 
         fontname='DecimaMonoPro')

# banner background height parameters
pad = 2 # points
bb= ax.get_window_extent()
h = bb.height/fig.dpi
height = ((banner.get_size()+2*pad)/72.)/h

# banner background
rect = plt.Rectangle((0,0), 
                     width=1, 
                     height=height,
                     transform=fig.transFigure, 
                     zorder=3,
                     fill=True, 
                     facecolor="grey", 
                     clip_on=False)
ax.add_patch(rect)

#transform coordinate of left
display_left_tuple = xfig_trans.transform((left,0))
display_left = display_left_tuple[0]

# shift title
title_shift = math.floor(tick_label_size * 2.6)
title_shift += title_pad

# title
graph.text(x = display_left - title_shift, y = 0.9, 
           transform = yfig_trans,
           s = title_text,
           fontproperties = prop2,
           weight = 'bold', 
           fontsize = 28,
           alpha = text_opacity)

# subtitle, +1 accounts for font size difference in title and subtitle
graph.text(x = display_left - title_shift + 1, y = 0.84, 
           transform = yfig_trans,
           s = subtitle_text,
           fontproperties=prop3,
           fontsize = 15, 
           alpha = text_opacity)

# adjust position of subplot in figure
plt.subplots_adjust(top=top)
plt.subplots_adjust(bottom=bottom)
plt.subplots_adjust(left=left)
plt.subplots_adjust(right=right)

# save to .svg
plt.savefig("test_chart.svg", dpi=300)


