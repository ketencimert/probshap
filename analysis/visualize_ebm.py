# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 00:50:42 2023

@author: Mert
"""

def make_plot_ebm(data_dict, feature_name, model_name, dataset_name, num_epochs='', debug=False):
    x_vals = data_dict["names"].copy()
    y_vals = data_dict["scores"].copy()

    # This is important since you do not plot plt.stairs with len(edges) == len(vals) + 1, which will have a drop to zero at the end
    y_vals = np.r_[y_vals, y_vals[np.newaxis, -1]] 

    # This is the code interpretml also uses: https://github.com/interpretml/interpret/blob/2327384678bd365b2c22e014f8591e6ea656263a/python/interpret-core/interpret/visual/plot.py#L115

    # main_line = go.Scatter(
    #     x=x_vals,
    #     y=y_vals,
    #     name="Main",
    #     mode="lines",
    #     line=dict(color="rgb(31, 119, 180)", shape="hv"),
    #     fillcolor="rgba(68, 68, 68, 0.15)",
    #     fill="none",
    # )
    #
    # main_fig = go.Figure(data=[main_line])
    # main_fig.show()
    # main_fig.write_image(f'plots/{model_name}_{dataset_name}_shape_{feature_name}_{num_epochs}epochs.pdf')
    
    
    # This is my custom code used for plotting
    x = np.array(x_vals)
    # if debug:
    #     print("Num cols:", num_cols)
    # if feature_name in num_cols:
    #     if debug:
    #         print("Feature to scale back:", feature_name)
    #     x = scaler_dict[feature_name].inverse_transform(x.reshape(-1, 1)).squeeze()
    # else:
    #     if debug:
    #         print("Feature not to scale back:", feature_name)

    plt.step(x, y_vals, where="post", color='black')
    # plt.fill_between(x, lower_bounds, mean, color='gray')
    # plt.fill_between(x, mean, upper_bounds, color='gray')
    plt.xlabel(f'Feature value')
    plt.ylabel('Feature effect on model output')
    plt.title(f'Feature:{feature_name}')
    # plt.savefig(f'plots/{model_name}_{dataset_name}_shape_{feature_name}_{num_epochs}epochs.pdf')
    plt.show()

ebm_global = model.explain_global()

for i in range(len(ebm_global.data()['names'])):
    data_names = ebm_global.data()
    feature_name = data_names['names'][i]
    shape_data = ebm_global.data(i)

    # if len(shape_data['names']) == 2:
    #     pass
    #     # make_one_hot_plot(shape_data['scores'][0], shape_data['scores'][1], feature_name, model_name, dataset_name)
    # else:
    make_plot_ebm(shape_data, 'x1', 'ebm', 'synthetic1')

# feat_for_vis = dict()
# for i, n in enumerate(ebm_global.data()['names']):
#     feat_for_vis[n] = {'importance': ebm_global.data()['scores'][i]}
# feature_importance_visualize(feat_for_vis, save_png=True, folder='.', name='ebm_feat_imp')