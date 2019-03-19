import torch
import torch.nn.functional as F
import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from trainer.model import ClassifyTransformer
from trainer.config import ModelParams, DataPath
from trainer.utils import DataOperat

ids_size = ModelParams.ids_size
d_model = ModelParams.d_model
hidden = ModelParams.hidden_size
d_ff = ModelParams.d_ff
N = ModelParams.N
n_classes = ModelParams.n_classes
n_heads = ModelParams.n_heads
device = ModelParams.device
batch_size = ModelParams.batch_size
label_data_path = './data/tgt_wolf.csv'
input_data_path = './data/src_wolf_320_pad.csv'
save_model_path = './data/training_2000_00_00_00_00_classify.pt'

Threshold_enlarge = 1e-4

print('test')
model = ClassifyTransformer(ids_size, n_classes, d_model, d_ff, N, n_heads, device=device)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
data_set = DataOperat.create_data_set(input_data_path, label_data_path, device)
data_set = data_set[:64*1]

def creat_img(x):
    N = x.size(0)
    M = x.size(1)
    idx = [i for i in range(x.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx)
    invert_x = x.cpu().index_select(0, idx)
    dir_img = torch.stack([
        torch.full((N, M), 255),
        (1-invert_x)**5*255,
        (1-invert_x)**5*255,
        torch.full((N, M), 127+64+32),
    ], dim=-1).type(torch.uint8).numpy()
    return dir_img, M, N

def creat_img2(plus, minus):
    N = plus.size(0)
    M = plus.size(1)
    # idx = [i for i in range(plus.size(0)-1, -1, -1)]
    idx = [i for i in range(plus.size(0))]
    idx = torch.LongTensor(idx)
    invert_plus = plus.cpu().index_select(0, idx)
    invert_minus = minus.cpu().index_select(0, idx)
    dir_img = torch.stack([
        (1-invert_minus)**5*255,
        (1-invert_plus)**5*255,
        (1-invert_plus)**5*255,
        torch.full((N, M), 127+64),
    ], dim=-1).type(torch.uint8).numpy()
    return dir_img, M, N


grads = {}
def save_grad(name):
    ''' 
    use leaf_tensor.register_hook(save_grad('ID_name'))
        loss.backward()
    and print(grads['ID_name'])
    '''
    def hook(grad):
        grads[name] = grad
    return hook


loss_fn = torch.nn.NLLLoss(ignore_index=0)
correct = 0
total = 0
figure_list = []
for x, y in data_set[:10]:
    model.eval()
    output = model(x.unsqueeze(0))
    predict = F.softmax(output)
    loss = loss_fn(output.unsqueeze(0), y.view(1))
    _, pred_ids = torch.max(output.data, 0) # TODO 1
    # size of attn.data is (batch, head, classes, secense)
    # print((model.decoder.layers[0].src_attention.attn.data[:, :, 2, :]).long(), 555)
    # print((model.decoder.layers[2].src_attention.attn.data[:, :, 2, :]).long(), 555)
    # attn_data = model.decoder.layers[0].src_attention.attn.data[:, :, int(pred_ids), :].squeeze()
    # attn_data = model.decoder.layers[0].src_attention.attn.data[:, 0, :, :].squeeze()
    attn_map = model.decoder.layers[1].src_attention.attn.squeeze()
    # attn_map = model.decoder.layers[1].src_attention.attn.squeeze()
    attn_data = torch.mean(attn_map, dim=0)
    
    model.decoder.layers[1].src_attention.attn.register_hook(save_grad('temp'))
    # model.decoder.layers[1].src_attention.attn.register_hook(save_grad('temp'))
    model.zero_grad()
    gradients = torch.full((output.size(0),), 0).to(device)
    # gradients = torch.tensor([0., 0., 1., 0., 0., 0.]).to(device)
    gradients[pred_ids] = 1
    output.backward(gradients)
    # grad_plan = (grads['temp'].squeeze()).clamp(min=0)

    #########################
    grad_plan = (grads['temp'].squeeze()).clamp(min=0)
    grad_plus = grad_plan * attn_map
    grad_flat = grad_plus.view(grad_plus.size(0), -1)
    grad_flat = grad_flat - torch.min(grad_flat, dim=-1, keepdim=True)[0]
    grad_flat = grad_flat / torch.max(grad_flat, dim=-1, keepdim=True)[0]
    grad_plus = grad_flat.view(grad_plus.size())
    attn_plus = torch.mean(grad_plus, dim=0)
    
    grad_plan = (grads['temp'].squeeze()*-1).clamp(min=0)
    grad_plus = grad_plan * attn_map
    grad_flat = grad_plus.view(grad_plus.size(0), -1)
    grad_flat = grad_flat - torch.min(grad_flat, dim=-1, keepdim=True)[0]
    grad_flat = grad_flat / torch.max(grad_flat, dim=-1, keepdim=True)[0]
    grad_plus = grad_flat.view(grad_plus.size())
    attn_minus = torch.mean(grad_plus, dim=0)
    ##########################

    grad_attn = (grads['temp'].squeeze()).clamp(min=0)


    # attn_data = torch.mean(grad_plus, dim=0)
    # attn_plus = torch.mean(attn_map*grad_plus, dim=0)
    
    # grad_plus = (1 - grad_plan) * attn_map
    # grad_flat = grad_plus.view(grad_plus.size(0), -1)
    # grad_flat = grad_flat - torch.min(grad_flat, dim=-1, keepdim=True)[0]
    # grad_flat = grad_flat / torch.max(grad_flat, dim=-1, keepdim=True)[0]
    # grad_plus = grad_flat.view(grad_plus.size())

    # attn_minus = torch.mean(attn_map*(1-grad_plus), dim=0)
    # attn_minus = torch.mean(attn_map - grad_plus, dim=0)




    # attn_data = torch.mean((F.softmax(grad_attn, dim=-1))*attn_map, dim=0)
    # attn_plus = torch.mean((F.softmax(grad_attn, dim=-1))*attn_map, dim=0)
    # # attn_data = torch.mean((1 - F.softmax(grad_attn, dim=-1))*attn_map, dim=0)
    # attn_minus = torch.mean((1 - F.softmax(grad_attn, dim=-1))*attn_map, dim=0)
    # # attn_data = torch.mean(F.softmax(grad_attn, dim=-1), dim=0)

    # attn_data = torch.mean(grad_attn, dim=0)
    # attn_minus = torch.mean(1-grad_attn, dim=0)
    # print(attn_data)

    # my_palette = bokeh.palettes.Category20c[20]
    # attn_i = grad_attn[:, 1:, :].transpose(0, 1)
    # for attn_j in attn_i:
    #     f = figure()
    #     for i, j in enumerate(attn_j):
    #         # j = F.softmax(j, dim=-1)
    #         f.line(range(len(j)), j.cpu().detach().numpy(), line_color=my_palette[i])
    #     show(f)
    
    # attn_zero = torch.tensor([[0. for i in range(255)]]).transpose(0, 1)
    # attn_data = reversed(attn_data.transpose(0, 1))
    # dir_img, X, Y = creat_img(attn_data)
    # dir_img, X, Y = creat_img2(attn_zero, attn_data)


    # for attention
    # dir_img, X, Y = creat_img(attn_data[1:])

    # for my_proposal
    dir_img, X, Y = creat_img2(attn_plus[1:], attn_minus[1:])
    
    figure_list.append(figure(plot_width=640, plot_height=320, x_range=(0, X), y_range=(0+1, Y+1),
                              title="{},{},{},{}".format(x.to('cpu')[:5], int(y), int(pred_ids),
                              (output[1:] *10).round().to('cpu').int().detach().numpy())
                              ))
    figure_list[total].image_rgba(image=[dir_img], x=[0], y=[0+1], dw=[X], dh=[Y])
    # figure_list.append(figure(plot_width=640, plot_height=200, x_range=(0, X), y_range=(Y+1, 0+1),
    #                           title="{},{},{},{}".format(x.to('cpu')[:5], int(y), int(pred_ids), output[1:].to('cpu').int().detach().numpy())
    #                           ))
    # figure_list[total].image_rgba(image=[dir_img], x=[0], y=[Y+1], dw=[X], dh=[Y])
    figure_list[total].xgrid.minor_grid_line_color = 'navy'
    figure_list[total].xgrid.minor_grid_line_alpha = 0.1
    figure_list[total].xaxis.axis_label = '会話の長さ'
    figure_list[total].yaxis.axis_label = 'クラスID'

    total += 1
    correct += 1 if pred_ids == y else 0
    print(x.cpu().detach())
    print(output.cpu().detach()[1:])
    print('true: ',int(y),'predict: ', int(pred_ids))
    print()

show(gridplot(figure_list, ncols=1))
print('Accuracy of all : {}'.format(correct / total))

class_correct = list(0. for i in range(n_classes))
class_total = list(0. for i in range(n_classes))
for x, y in data_set:
    output = model(x.unsqueeze(0))
    _, pred_ids = torch.max(output.data, 0) # TODO 1
    correct_ = 1 if int(pred_ids) == int(y) else 0
    for i in range(n_classes):
        y_idx = y
        class_correct[y_idx] += correct_
        class_total[y_idx] += 1

for i in range(n_classes):
    if class_correct[i] < 1 or class_total[i] < 1:
        continue
    print('Accuracy of {} : {}'.format(i, class_correct[i]/class_total[i]))
