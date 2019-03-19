from bokeh.plotting import figure, output_file, show
from trainer.utils import DataOperat

train_acc = []
val_acc = []
train_loss = []
val_loss = []
x = []

# prepare some data
for i, line in enumerate(DataOperat.load_csv(
    './data/training_2000_00_00_00_00_losses.csv')):
    print(line)
    train_acc.append(float(line[0]))
    val_acc.append(float(line[1]))
    train_loss.append(float(line[2]))
    val_loss.append(float(line[3]))
    x.append(i)

# output to static HTML file
output_file("log_lines.html", mode='inline')

# create a new plot
p = figure(
    tools="pan,box_zoom,wheel_zoom,xwheel_zoom,ywheel_zoom,reset,undo,save,hover"
)

# add some renderers
p.line(x, train_acc, legend="train_accuracy", line_color="red", line_width=4)
p.line(x, val_acc, legend="val_accuracy", line_color="blue", line_width=4)
p.line(x, val_loss, legend="validation loss", line_color="green", line_width=8)
p.line(x, train_loss, legend="train loss", line_color="black", line_width=3)
# p.legend.location = "top_right"
p.xaxis.axis_label = 'epoch'
p.xaxis.major_label_text_font_size = '24pt'
p.yaxis.major_label_text_font_size = '24pt'
# p.legend.label_text_font_size = '24pt'
p.xaxis.axis_label_text_font_size = '24pt'

# show the results
show(p)
