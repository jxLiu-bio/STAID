import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as ss
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from plotnine import *


def scatter_gene_distri(adata,
                        gene,
                        size=3,
                        shape='h',
                        cmap='magma',
                        spatial_info=['array_row', 'array_col'],
                        coord_ratio=0.7,
                        return_plot=False):
    if gene in adata.obs.columns:
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata.obs.loc[:, gene].values,
                                   index=adata.obs_names,
                                   columns=[gene])
        else:
            plot_df = pd.DataFrame(adata.obs.loc[:, gene],
                                   index=adata.obs_names,
                                   columns=gene)
        if spatial_info in adata.obsm_keys():
            plot_df['x'] = adata.obsm[spatial_info][:, 0]
            plot_df['y'] = adata.obsm[spatial_info][:, 1]
        elif set(spatial_info) <= set(adata.obs.columns):
            plot_coor = adata.obs
            plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
            plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values

        if isinstance(gene, str):
            base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y', fill=gene),
                                               shape=shape, stroke=0.1, size=size) +
                         xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                         ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                         scale_fill_cmap(cmap_name=cmap) +
                         coord_equal(ratio=coord_ratio) +
                         theme_classic() +
                         theme(legend_position=('right'),
                               legend_background=element_blank(),
                               legend_key_width=4,
                               legend_key_height=50)
                         )
            print(base_plot)
        else:
            for i in gene:
                base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y', fill=gene),
                                                   shape=shape, stroke=0.1, size=size) +
                             xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                             ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                             scale_fill_cmap(cmap_name=cmap) +
                             coord_equal(ratio=coord_ratio) +
                             theme_classic() +
                             theme(legend_position=('right'),
                                   legend_background=element_blank(),
                                   legend_key_width=4,
                                   legend_key_height=50)
                             )
                print(base_plot)

        return
    if ss.issparse(adata.X):
        plot_df = pd.DataFrame(adata.X.todense(), index=adata.obs_names,
                               columns=adata.var_names)
    else:
        plot_df = pd.DataFrame(adata.X, index=adata.obs_names,
                               columns=adata.var_names)
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 0]
        plot_df['y'] = adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
    plot_df['radius'] = size
    plot_df = plot_df.sort_values(by=gene, ascending=True)
    if isinstance(gene, str):
        base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y', fill=gene),
                                           shape=shape, stroke=0.1, size=size) +
                     xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                     ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                     scale_fill_cmap(cmap_name=cmap) +
                     coord_equal(ratio=coord_ratio) +
                     theme_classic() +
                     theme(legend_position=('right'),
                           legend_background=element_blank(),
                           legend_key_width=4,
                           legend_key_height=50)
                     )
        print(base_plot)
    else:
        for i in gene:
            base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y', fill=gene),
                                               shape=shape, stroke=0.1, size=size) +
                         xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                         ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                         scale_fill_cmap(cmap_name=cmap) +
                         coord_equal(ratio=coord_ratio) +
                         theme_classic() +
                         theme(legend_position=('right'),
                               legend_background=element_blank(),
                               legend_key_width=4,
                               legend_key_height=50)
                         )
            print(base_plot)
    if return_plot:
        return base_plot


def scatter_tm_binary(adata, tm, size=3, shape='h',
                      spatial_info=['array_row', 'array_col'],
                      colors=['#CA1C1C', '#CCCCCC'],
                      coord_ratio=0.7, return_plot=False):
    if '-' in tm:
        tm = 'tm-' + tm.split('-')[0] + "_subTm-" + tm.split('-')[1]
        plot_df = adata.obsm['subTm_binary']
    else:
        tm = 'tm_' + tm
        plot_df = adata.obsm['tm_binary']
    plot_df = pd.DataFrame(plot_df)
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 0]
        plot_df['y'] = adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[tm]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
    plot_df['radius'] = size
    plot_df[tm] = plot_df[tm].values.astype(int)
    plot_df[tm] = plot_df[tm].values.astype(str)
    plot_df[tm] = pd.Categorical(plot_df[tm],
                                 categories=['1', '0'],
                                 ordered=True)
    base_plot = (ggplot() + geom_point(plot_df, aes(x='x', y='y', fill=tm),
                                       shape=shape, stroke=0.1, size=size) +
                 xlim(min(plot_df.x) - 1, max(plot_df.x) + 1) +
                 ylim(min(plot_df.y) - 1, max(plot_df.y) + 1) +
                 scale_fill_manual(values=colors) +
                 coord_equal(ratio=coord_ratio) +
                 theme_classic() +
                 theme(legend_position=('right'),
                       legend_background=element_blank(),
                       legend_key_width=4,
                       legend_key_height=50)
                 )
    print(base_plot)
    if return_plot:
        return base_plot


# tissue module id card
def tm_heatmap_signal_tm_id_card(adata,
                                 tm,
                                 domain='freq_domain_svg', figsize=(6, 2),
                                 dpi=100, color='#CA1C1C',
                                 y_range=[0, 0.08],
                                 return_fig=False, ax=None, title=None, **kwargs):
    freq_signal = \
        adata.uns['detect_TM_data']['freq_signal_tm'].loc[tm, :].values.reshape(1, -1)
    # print(freq_signal)
    if title != None:
        plt.title(title, y=-0.5)
    ax = sns.heatmap(data=freq_signal, cbar=False, cmap="Reds")
    ax.tick_params(left=False, bottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


# tissue module id card
def cell_type_proportion_box_tm_id_card(cell_type_name, cell_type_proportion_data, ax=None, title=None):
    boxplot_data = []
    for val in cell_type_name:
        boxplot_data.append(cell_type_proportion_data[val])
    # plt.title("Cell type proportion")
    labels = [x.replace("q05cell_abundance_w_sf_", "") for x in cell_type_name]
    plt.title(title)
    # plt.subplots_adjust(left=0.8)
    ax.boxplot(boxplot_data, labels=labels, showfliers=False, patch_artist=True, boxprops={"facecolor": "#FF2A6B"},
               medianprops={"color": "black"})
    ax.yaxis.set_tick_params(labelsize=7, )
    ax.xaxis.set_tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)


# tissue module id card
def tm_spatial_map_scatter_tm_id_card(adata, tm, tm_color, title, radius=0.5, spatial_info=['array_row', 'array_col'],
                                      ax=None):
    x = []
    y = []
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 0]
        y = adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    color = []
    for index in range(len(tm)):
        if tm[index][0] == "1":
            color.append(tm_color)
        else:
            color.append('gray')
    if title != None:
        plt.title(title, y=-0.5)
    ax.scatter(max(y) - y, max(x) - x, s=radius, c=color)
    ax.minorticks_on()
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.grid(False)


# tissue module id card
def scatter_SVGs_distri_tm_id_card(adata,
                                   gene,
                                   size=0.03,
                                   shape='h',
                                   cmap='magma',
                                   spatial_info=['array_row', 'array_col'],
                                   ax=None,
                                   coord_ratio=1,
                                   return_plot=False):
    if gene in adata.obs.columns:
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata.obs.loc[:, gene].values,
                                   index=adata.obs_names,
                                   columns=[gene])
        else:
            plot_df = pd.DataFrame(adata.obs.loc[:, gene],
                                   index=adata.obs_names,
                                   columns=gene)
        if spatial_info in adata.obsm_keys():
            plot_df['x'] = adata.obsm[spatial_info][:, 0]
            plot_df['y'] = adata.obsm[spatial_info][:, 1]
        elif set(spatial_info) <= set(adata.obs.columns):
            plot_coor = adata.obs
            plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
            plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        if isinstance(gene, str):
            return plot_scatter_tm_id_card(max(plot_df.y) - plot_df.y, max(plot_df.x) - plot_df.x, plot_df[gene],
                                           gene, cmap, plot_df['radius'], a=ax)
        return
    if ss.issparse(adata.X):
        plot_df = pd.DataFrame(adata.X.todense(), index=adata.obs_names,
                               columns=adata.var_names)
    else:
        plot_df = pd.DataFrame(adata.X, index=adata.obs_names,
                               columns=adata.var_names)
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 0]
        plot_df['y'] = adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        plot_df['radius'] = size
        plot_df = plot_df.sort_values(by=gene, ascending=True)
        print(plot_df)
        if isinstance(gene, str):
            return plot_scatter_tm_id_card(max(plot_df.y) - plot_df.y, max(plot_df.x) - plot_df.x, plot_df[gene], gene,
                                           cmap,
                                           plot_df['radius'], ax=ax)
        else:
            return


def plot_scatter_tm_id_card(x, y, colors, title, cmap, marker='h',
                            radius=None, ax=None, up_title=False):
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.9)
    if up_title:
        plt.title(title)
    else:
        plt.title(title, y=-0.5)
    if isinstance(radius, int) or isinstance(radius, float):
        scatter = ax.scatter(x, y, s=radius, marker=marker, c=colors, cmap=cmap)
    else:
        scatter = ax.scatter(x, y, c=colors, marker=marker, cmap=cmap)
    ax.minorticks_on()
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')
    ax.grid(False)
    return scatter


def scatter_gene(adata,
                 gene,
                 size=None,
                 shape='h',
                 cmap='magma',
                 spatial_info=['array_row', 'array_col'],
                 coord_ratio=1,
                 return_fig=False, save_path=None):
    if isinstance(gene, np.ndarray):
        gene = list(gene)
    if isinstance(gene, pd.core.indexes.base.Index):
        gene = list(gene)
    if ss.issparse(adata.X):
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    else:
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 0]
        plot_df['y'] = adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        # print(plot_df)
    if isinstance(gene, str):
        fig, ax = plt.subplots()
        if size == None:
            scatter = plot_scatter_tm_id_card(x=plot_df['x'],
                                              y=plot_df['y'],
                                              colors=plot_df[gene],
                                              title=gene,
                                              marker=shape,
                                              cmap=cmap,
                                              ax=ax,
                                              up_title=True)
            plt.colorbar(scatter, ax=ax)
        elif isinstance(size, int) or isinstance(size, float):
            scatter = plot_scatter_tm_id_card(x=plot_df['x'],
                                              y=plot_df['y'],
                                              colors=plot_df[gene],
                                              title=gene,
                                              marker=shape,
                                              cmap=cmap,
                                              radius=size,
                                              ax=ax,
                                              up_title=True)
            plt.colorbar(scatter, ax=ax)
        plt.show()
        if save_path != None:
            plt.savefig(save_path)
        if return_fig:
            return ax

    elif isinstance(gene, list) or isinstance(gene, np.ndarray):
        row = math.ceil(len(gene) / 4)
        fig = plt.figure(dpi=350,
                         constrained_layout=True,
                         figsize=(20, row * 5)
                         )

        gs = GridSpec(row, 4,
                      figure=fig)
        print(row, 4)
        ax_list = []
        for index, value in enumerate(gene):
            ax = fig.add_subplot(gs[index // 4, index % 4])

            if size == None:
                scatter = plot_scatter_tm_id_card(x=plot_df['x'],
                                                  y=plot_df['y'],
                                                  colors=plot_df[value],
                                                  title=value,
                                                  cmap=cmap,
                                                  marker=shape,
                                                  ax=ax,
                                                  up_title=True)
                plt.colorbar(scatter, ax=ax)
            elif isinstance(size, int) or isinstance(size, float):
                scatter = plot_scatter_tm_id_card(x=plot_df['x'],
                                                  y=plot_df['y'],
                                                  colors=plot_df[value],
                                                  title=value,
                                                  cmap=cmap,
                                                  marker=shape,
                                                  radius=size,
                                                  ax=ax,
                                                  up_title=True)
                plt.colorbar(scatter, ax=ax)
            ax_list.append(ax)

        if save_path:
            plt.savefig(save_path)
        plt.show()
        if return_fig:
            return ax_list


def scatter_tm_expression(adata,
                          tm,
                          cmap='magma',
                          radius=None,
                          spatial_info=['array_row', 'array_col'],
                          save_path=None,
                          return_fig=False):
    x = []
    y = []
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    if isinstance(tm, str):
        tm_value = adata.obsm["tm_pseudo_expression"][tm].values
        fig, ax = plt.subplots()
        plt.title(tm)
        if radius != None:
            scatter = ax.scatter(y, max(x) - x, s=radius, c=tm_value, cmap=cmap)
        else:
            scatter = ax.scatter(y, max(x) - x, c=tm_value, cmap=cmap)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(labelsize=10)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
        ax.grid(False)

        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax
    elif isinstance(tm, list) or isinstance(tm, np.ndarray):
        row = math.ceil(len(tm) / 4)
        fig = plt.figure(dpi=350,
                         constrained_layout=True,  # 类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
                         figsize=(20, row * 5)
                         )

        gs = GridSpec(row, 4,
                      figure=fig)
        # print(row, 4)
        ax_list = []
        for index, value in enumerate(tm):
            ax = fig.add_subplot(gs[index // 4, index % 4])
            tm_value = adata.obsm["tm_pseudo_expression"][value].values

            plt.title(value)
            if radius != None:
                scatter = ax.scatter(y, max(x) - x, s=radius, c=tm_value, cmap=cmap)
            else:
                scatter = ax.scatter(y, max(x) - x, c=tm_value, cmap=cmap)
            ax.minorticks_on()
            ax.yaxis.set_tick_params(labelsize=10)
            ax.xaxis.set_tick_params(labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax)
            ax.grid(False)
            ax_list.append(ax)
        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax_list


def scatter_umap_clustering(adata, svg_list, save_path=None, return_fig=False):
    current_genes = adata.uns['detect_TM_data']['gft_umap_tm'].index.tolist()
    if set(svg_list) <= set(current_genes):
        svg_list = np.intersect1d(svg_list, current_genes)
    else:
        diff_genes = np.setdiff1d(svg_list, current_genes)
        raise KeyError(f"{diff_genes} are not calculated in the above step.")
    plot_df = pd.concat((adata.uns['detect_TM_data']['gft_umap_tm'].loc[svg_list, :],
                         adata.var.loc[svg_list, :].tissue_module), axis=1)

    categories = [eval(i) for i in np.unique(plot_df.tissue_module)]
    categories = np.sort(np.array(categories))
    categories = categories.astype(str)
    plot_df.tissue_module = pd.Categorical(plot_df.tissue_module,
                                           categories=categories)
    base_plot = (ggplot(plot_df,
                        aes('UMAP_1', 'UMAP_2',
                            fill='tissue_module'))
                 + geom_point(size=4)
                 + theme_classic())
    print(base_plot)

    if save_path != None:
        base_plot.save(f"{save_path}")
    if return_fig:
        return base_plot


def scatter_tm_gene(adata,
                    tm,
                    gene,  # list
                    cmap="magma",
                    tm_color='#B4671F',
                    radius=None,
                    spatial_info=['array_row', 'array_col'],
                    save_path=None,
                    return_fig=False):
    if isinstance(gene, str):
        gene = [gene]
    if isinstance(gene, pd.core.indexes.base.Index):
        gene = list(gene)
    x = []
    y = []
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    row = math.ceil((1 + len(gene)) / 4)
    fig = plt.figure(dpi=350,
                     constrained_layout=True,  # 类似于tight_layout，使得各子图之间的距离自动调整【类似excel中行宽根据内容自适应】
                     figsize=(20, row * 5)
                     )

    gs = GridSpec(row, 4,
                  figure=fig)
    ax_list = []
    ###########################################################
    ax_tm = fig.add_subplot(gs[0, 0])
    tm_value = [int(x) for x in list(adata.obsm["tm_binary"][tm].values)]
    cmap_tm = ListedColormap(["#b4b4b4", tm_color])

    plt.title(tm)
    if radius != None:
        scatter = ax_tm.scatter(y, max(x) - x, s=radius, c=tm_value, cmap=cmap_tm)
    else:
        scatter = ax_tm.scatter(y, max(x) - x, c=tm_value, cmap=cmap_tm)

    ax_tm.minorticks_on()
    ax_tm.yaxis.set_tick_params(labelsize=10)
    ax_tm.xaxis.set_tick_params(labelsize=10)
    ax_tm.spines['top'].set_visible(False)
    ax_tm.spines['right'].set_visible(False)
    ax_tm.spines['left'].set_visible(False)
    ax_tm.spines['bottom'].set_visible(False)
    ax_tm.get_xaxis().set_visible(False)
    ax_tm.get_yaxis().set_visible(False)
    ax_tm.set_aspect('equal')
    ax_tm.grid(False)
    plt.legend(*scatter.legend_elements(), loc="center right", bbox_to_anchor=(1, 0, 0.15, 1))
    ax_list.append(ax_tm)
    #########################

    if isinstance(gene, np.ndarray):
        gene = list(gene)
    if ss.issparse(adata.X):
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X.todense(),
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    else:
        if isinstance(gene, str):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=[gene])
        elif isinstance(gene, list) or isinstance(gene, np.ndarray):
            plot_df = pd.DataFrame(adata[:, gene].X,
                                   index=adata.obs_names,
                                   columns=gene)
        else:
            raise KeyError(f"{gene} is invalid!")
    if spatial_info in adata.obsm_keys():
        plot_df['x'] = adata.obsm[spatial_info][:, 1]
        plot_df['y'] = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        plot_df = plot_df[gene]
        plot_df = pd.DataFrame(plot_df)
        plot_df['x'] = plot_coor.loc[:, spatial_info[0]].values
        plot_df['y'] = plot_coor.loc[:, spatial_info[1]].values
        # print(plot_df)
    if isinstance(gene, list) or isinstance(gene, np.ndarray):
        for index, value in enumerate(gene):
            ax = fig.add_subplot(gs[(1 + index) // 4, (1 + index) % 4])

            if radius == None:
                scatter = plot_scatter_tm_id_card(x=plot_df.y,
                                                  y=max(plot_df.x) - plot_df.x,
                                                  colors=plot_df[value],
                                                  title=value,
                                                  cmap=cmap,
                                                  ax=ax,
                                                  up_title=True)
            elif isinstance(radius, int) or isinstance(radius, float):
                scatter = plot_scatter_tm_id_card(x=plot_df.y,
                                                  y=max(plot_df.x) - plot_df.x,
                                                  colors=plot_df[value],
                                                  title=value,
                                                  cmap=cmap,
                                                  radius=radius,
                                                  ax=ax,
                                                  up_title=True)
            plt.colorbar(scatter, ax=ax)
            ax_list.append(ax)
    plt.show()
    if save_path:
        plt.savefig(save_path)
    if return_fig:
        return ax_list


def scatter_cell_type(adata,
                      cell_type,
                      cmap='magma',
                      size=None,
                      shape='s',
                      spatial_info=['x', 'y'],
                      prop_key='deconvolution',
                      vmax=None,
                      n_col=4,
                      save_path=None,
                      return_fig=False,
                      fig_size=1,
                      dpi=100):
    x = []
    y = []
    if isinstance(cell_type, pd.core.indexes.base.Index):
        cell_type = list(cell_type)
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 0]
        y = -1 * adata.obsm[spatial_info][:, 1]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values

    if isinstance(cell_type, str):
        if vmax == 'auto':
            vmax = adata.obsm[prop_key].loc[:,
                   [cell_type]].values.flatten().max()
            vmax_s = vmax
        ct_value = adata.obsm[prop_key][cell_type].values
        fig, ax = plt.subplots()
        plt.title(cell_type)
        if vmax == None:
            vmax_s = max(ct_value)
        elif isinstance(vmax, str):
            vmax_s = np.percentile(ct_value, float(vmax[1:]))
        else:
            vmax_s = vmax
        if size != None:
            scatter = ax.scatter(x, y, s=size, c=ct_value, cmap=cmap,
                                 vmax=vmax_s, marker=shape)
        else:
            scatter = ax.scatter(x, y, c=ct_value, cmap=cmap,
                                 vmax=vmax_s, marker=shape)
        ax.minorticks_on()
        # ax.yaxis.set_tick_params(labelsize=10)
        # ax.xaxis.set_tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, aspect=30, pad=0.02)
        ax.grid(False)
        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax
    elif isinstance(cell_type, list) or isinstance(cell_type, np.ndarray):
        if vmax == 'auto':
            vmax = adata.obsm[prop_key].loc[:, cell_type].values.max()
        row = math.ceil(len(cell_type) / n_col)
        fig = plt.figure(dpi=dpi,
                         constrained_layout=True,
                         figsize=(n_col * fig_size, row * fig_size)
                         )

        gs = GridSpec(row, n_col,
                      figure=fig)
        ax_list = []
        if vmax == None:
            for index, value in enumerate(cell_type):
                ax = fig.add_subplot(gs[index // n_col, index % n_col])
                ct_value = adata.obsm[prop_key][value].values

                plt.title(value)
                if size != None:
                    scatter = ax.scatter(x, y, s=size, c=ct_value, cmap=cmap,
                                         marker=shape)
                else:
                    scatter = ax.scatter(x, y, c=ct_value, cmap=cmap,
                                         marker=shape)
                ax.minorticks_on()
                # ax.yaxis.set_tick_params(labelsize=10)
                # ax.xaxis.set_tick_params(labelsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect('equal')
                plt.colorbar(scatter, ax=ax, aspect=30, pad=0.02)
                ax.grid(False)
                ax_list.append(ax)
        elif isinstance(vmax, str):
            if vmax[0] != 'p':
                raise ValueError("If vmax is str but not 'auto',\
                                 it should be pXX, e.g. p99")
            for index, value in enumerate(cell_type):
                ax = fig.add_subplot(gs[index // n_col, index % n_col])
                ct_value = adata.obsm[prop_key][value].values
                vmax_s = np.percentile(ct_value, float(vmax[1:]))
                plt.title(value)
                if size != None:
                    scatter = ax.scatter(x, y, s=size, c=ct_value, cmap=cmap,
                                         vmax=vmax_s, marker=shape)
                else:
                    scatter = ax.scatter(x, y, c=ct_value, cmap=cmap,
                                         vmax=vmax_s, marker=shape)
                ax.minorticks_on()
                ax.yaxis.set_tick_params(labelsize=10)
                ax.xaxis.set_tick_params(labelsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect('equal')
                plt.colorbar(scatter, ax=ax, aspect=30, pad=0.02)
                ax.grid(False)
                ax_list.append(ax)
        else:
            for index, value in enumerate(cell_type):
                ax = fig.add_subplot(gs[index // n_col, index % n_col])
                ct_value = adata.obsm[prop_key][value].values
                plt.title(value)

                if size != None:
                    scatter = ax.scatter(x, y, s=size, c=ct_value, cmap=cmap,
                                         vmax=vmax, marker=shape)
                else:
                    scatter = ax.scatter(x, y, c=ct_value, cmap=cmap,
                                         vmax=vmax, marker=shape)
                ax.minorticks_on()
                # ax.yaxis.set_tick_params(labelsize=10)
                # ax.xaxis.set_tick_params(labelsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_aspect('equal')
                if (index % n_col) == (n_col - 1) or \
                        index == len(cell_type) - 1:
                    plt.colorbar(scatter, ax=ax, aspect=30, pad=0.02)
                ax.grid(False)
                ax_list.append(ax)
        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax_list


def add_image(adata,
              spatial_dir,
              library_id=None):
    from matplotlib.image import imread
    import json
    from pathlib import Path
    if library_id == None:
        library_id = 'spatial_dataset'
    path = Path(spatial_dir)
    adata.uns["spatial"] = dict()
    library_id = 'spatial'
    adata.uns['spatial'][library_id] = {}
    load_images = True
    if load_images:
        files = dict(
            tissue_positions_file=path / 'tissue_positions_list.csv',
            scalefactors_json_file=path / 'scalefactors_json.json',
            hires_image=path / 'tissue_hires_image.png',
            lowres_image=path / 'tissue_lowres_image.png',
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    print(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]['images'] = dict()
        for res in ['hires', 'lowres']:
            try:
                adata.uns["spatial"][library_id]['images'][res] = imread(
                    str(files[f'{res}_image'])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]['scalefactors'] = json.loads(
            files['scalefactors_json_file'].read_bytes()
        )

        # read coordinates
        positions = pd.read_csv(files['tissue_positions_file'], header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']
        adata.obs.drop(np.intersect1d(positions.columns,
                                      adata.obs.columns),
                       inplace=True,
                       axis=1)
        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm['spatial'] = adata.obs[
            ['pxl_row_in_fullres', 'pxl_col_in_fullres']
        ].to_numpy()
        adata.obs.drop(
            columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
            inplace=True,
        )
    return adata


def pie_spot(loc_list, frac_list, size, color_list):
    frac_cumsum = np.cumsum(frac_list)
    frac_cumsum = frac_cumsum / frac_cumsum[-1]
    marker_list = []
    previous = 0
    # calculate the points of the pie pieces
    for color, frac in zip(color_list, frac_cumsum):
        curr = frac
        x = np.cos(2 * np.pi * np.linspace(previous, curr, 50)).tolist()
        y = np.sin(2 * np.pi * np.linspace(previous, curr, 50)).tolist()
        xy = np.row_stack([[0, 0], np.column_stack([x, y])])
        marker_list.append({'marker': xy, 's': size * np.abs(xy).max() ** 2,
                            'facecolor': color, 'edgecolor': "darkgrey",
                            'linewidth': 0.1})
        previous = frac
    # scatter each of the pie pieces to create pies
    point_marker_list = []
    for marker in marker_list:
        point_marker_list.append(loc_list + [marker])
    return (point_marker_list)


def scatter_pie(adata,
                spatial_key=['x', 'y'],
                deconv_key='deconvolution',
                pt_size=8,
                figsize=(8, 6),
                title=None,
                colors=None,
                show_legend=True,
                return_fig=False,
                save_path=None):
    decon_df = adata.obsm[deconv_key]
    if isinstance(spatial_key, str):
        loc_df = pd.DataFrame(adata.obsm[spatial_key],
                              index=adata.obs_names,
                              columns=['x', 'y'])
        spatial_key = ['x', 'y']
    elif set(spatial_key) < set(adata.obs.columns):
        loc_df = adata.obs.loc[:, spatial_key]
    x_max, y_max = loc_df.max(axis=0)
    x_min, y_min = loc_df.min(axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_max = x_max + 0.02 * x_range
    x_min = x_min - 0.02 * x_range
    y_max = y_max + 0.02 * y_range
    y_min = y_min - 0.02 * y_range
    if colors:
        color_pal = colors
    else:
        color_pal = sns.color_palette("Paired_r", len(decon_df.columns))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    for i in decon_df.index:
        deconv_list = decon_df.loc[i, :]
        loc_list = loc_df.loc[i, :].tolist()
        point_marker_list = pie_spot(loc_list[0:2],
                                     deconv_list,
                                     pt_size ** 2,
                                     color_pal)
        for point_marker in point_marker_list:
            ax.scatter(point_marker[0],
                       point_marker[1],
                       **point_marker[-1])
    # add legends
    cell_types = decon_df.columns
    patch_list = []
    for i in range(len(cell_types)):
        patch_list.append(mpatches.Patch(facecolor=color_pal[i],
                                         label=cell_types[i],
                                         edgecolor="darkgrey",
                                         linewidth=0.1))
    if show_legend:
        ax.legend(handles=patch_list,
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  frameon=False,
                  handlelength=1,
                  handleheight=1)
    ax.axis('equal')
    # ax.set_xlabel(loc_df.columns[0])
    # ax.set_ylabel(loc_df.columns[1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    if title:
        ax.set_title(title, pad=15)
    if return_fig:
        return ax
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def scatter_spots(adata,
                  feature,
                  size=None,
                  shape='s',
                  spatial_info=['x', 'y'],
                  cmap='tab20',
                  all_classes=None,
                  all_colors=None,
                  save_path=None,
                  return_fig=False,
                  n_col=4,
                  show_legend=True,
                  ):
    x = []
    y = []
    if spatial_info in adata.obsm_keys():
        x = adata.obsm[spatial_info][:, 1]
        y = adata.obsm[spatial_info][:, 0]
    elif set(spatial_info) <= set(adata.obs.columns):
        plot_coor = adata.obs
        x = plot_coor.loc[:, spatial_info[0]].values
        y = plot_coor.loc[:, spatial_info[1]].values
    pos_df = pd.DataFrame([x, y], columns=adata.obs_names,
                          index=['x', 'y'])
    pos_df = pos_df.transpose()
    if isinstance(feature, str):
        tm_value = adata.obs[feature].values
        fig, ax = plt.subplots(figsize=(11, 5))
        plt.title(feature)
        if not all_classes:
            all_classes = np.unique(tm_value)
        if not all_colors:
            all_colors = plt.get_cmap(cmap, range(len(all_classes)))
        scatter_list = []
        for ind, cla in enumerate(all_classes):
            color = all_colors[ind]
            corresponding_spots = adata.obs_names[adata.obs[feature] == cla]
            x_tmp = pos_df.loc[corresponding_spots, 'x'].values
            y_tmp = pos_df.loc[corresponding_spots, 'y'].values
            if size != None:
                scatter = ax.scatter(x_tmp, y_tmp, s=size, c=color,
                                     marker=shape)
            else:
                scatter = ax.scatter(x_tmp, y_tmp, c=color, marker=shape)
            scatter_list.append(scatter)
        ax.minorticks_on()
        ax.yaxis.set_tick_params(labelsize=10)
        ax.xaxis.set_tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        ax.grid(False)
        if show_legend:
            plt.legend(scatter_list, list(all_classes),
                       bbox_to_anchor=(1, -0.05, 0.6, 1),
                       frameon=False)
        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax
    elif isinstance(feature, list):
        row = int(np.ceil(len(feature) / n_col))
        fig = plt.figure(dpi=350,
                         constrained_layout=True,
                         figsize=(n_col * 5, row * 5)
                         )
        # plt.rcParams['figure.figsize'] = (10, 8)

        gs = GridSpec(row,
                      n_col,
                      figure=fig)
        ax_list = []
        for index, value in enumerate(feature):
            ax = fig.add_subplot(gs[index // n_col, index % n_col])
            tm_value = adata.obs[value].values
            plt.title(value)
            if not all_classes:
                all_classes = np.unique(tm_value)
            if not all_colors:
                all_colors = plt.get_cmap(cmap, range(len(all_classes)))
            scatter_list = []
            for ind, cla in enumerate(all_classes):
                color = all_colors[ind]
                corresponding_spots = adata.obs_names[adata.obs[value] == cla]
                x_tmp = pos_df.loc[corresponding_spots, 'x'].values
                y_tmp = pos_df.loc[corresponding_spots, 'y'].values
                if size != None:
                    scatter = ax.scatter(x_tmp, y_tmp, s=size, c=color,
                                         marker=shape)
                else:
                    scatter = ax.scatter(x_tmp, y_tmp, c=color, marker=shape)
                scatter_list.append(scatter)
            ax.minorticks_on()
            ax.yaxis.set_tick_params(labelsize=10)
            ax.xaxis.set_tick_params(labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            ax.grid(False)
            ax_list.append(ax)
        if show_legend:
            if show_legend:
                plt.legend(scatter_list, list(all_classes),
                           bbox_to_anchor=(1, -0.2, 0.3, 1.15),
                           prop={'size': 11.6},
                           frameon=False)
        if save_path:
            plt.savefig(f"{save_path}")
        plt.show()
        if return_fig:
            return ax_list
