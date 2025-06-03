import torch as th
from torch_scatter import scatter_add
from ocnn.octree import Octree
from ocnn.utils import cumsum


def get_xyz_from_octree(octree, depth, nempty=False, return_batch=False):
    scale = 2 ** (1 - depth)
    x, y, z, b = octree.xyzb(depth=depth, nempty=nempty)
    xyz = (th.stack([x, y, z], dim=1) + 0.5) * scale - 1.0
    if return_batch:
        return xyz, b
    else:
        return xyz


def search_value(key: th.Tensor, query: th.Tensor, value: th.Tensor = None):
    r''' Searches values according to sorted shuffled keys.

    Args:
      value (th.Tensor): The input tensor with shape (N, C).
      key (th.Tensor): The key tensor corresponds to :attr:`value` with shape
          (N,), which contains sorted shuffled keys of an octree.
      query (th.Tensor): The query tensor, which also contains shuffled keys.
    '''

    # deal with out-of-bound queries, the indices of these queries
    # returned by th.searchsorted equal to `key.shape[0]`
    out_of_bound = query > key[-1]

    # search
    idx = th.searchsorted(key, query)
    idx[out_of_bound] = -1   # to avoid overflow when executing the following line
    found = key[idx] == query

    if value is not None:
        # assign the found value to the output
        out = th.zeros(query.shape[0], value.shape[1], device=value.device)
        out[found] = value[idx[found]]
        return out, found
    else:
        return idx, found


def octree_align(value: th.Tensor, octree: Octree, octree_query: Octree,
                 depth: int, nempty: bool = False):
    r''' Wraps :func:`octree_align` to take octrees as input for convenience.
    '''

    key = octree.key(depth, nempty)
    query = octree_query.key(depth, nempty)
    assert key.shape[0] == value.shape[0]
    return search_value(key, query, value)


def octree_align_with_map(value: th.Tensor, octree: Octree, octree_query: Octree, octree_map: th.Tensor,
                 depth: int, nempty: bool = False):
    r''' Wraps :func:`octree_align` to take octrees as input for convenience.
    '''
    # batch_nnum = octree_query.batch_nnum[depth].to(octree.device)
    batch_nnum = th.bincount(octree_query.batch_id(depth, nempty=nempty), minlength=octree_map.shape[0])
    key = octree.key(depth, nempty)
    query = octree_query.key(depth, nempty)
    fake_batch_id = th.repeat_interleave(octree_map, batch_nnum)
    query = (fake_batch_id << 48) | (query & ((1 << 48) - 1))
    assert key.shape[0] == value.shape[0]
    return search_value(key, query, value)


def octree_scatter_add_with_map(value: th.Tensor, octree: Octree, octree_query: Octree, octree_map: th.Tensor,
                                depth: int, nempty: bool = False):
    r''' Wraps :func:`octree_align` to take octrees as input for convenience.
    '''
    # batch_nnum = octree_query.batch_nnum[depth].to(octree.device)
    batch_nnum = th.bincount(octree_query.batch_id(depth, nempty=nempty), minlength=octree_map.shape[0])
    fake_batch_id = th.repeat_interleave(octree_map, batch_nnum)
    value = th.repeat_interleave(value, batch_nnum)
    key = octree.key(depth, nempty)
    query = octree_query.key(depth, nempty)
    query = (fake_batch_id << 48) | (query & ((1 << 48) - 1))
    assert query.shape[0] == value.shape[0]
    idx, _ = search_value(key, query)
    out = th.zeros_like(key)
    return scatter_add(value, idx, out=out)


def octree_search(octree: Octree, octree_query: Octree,
                  depth: int, nempty: bool = False):
    r''' Wraps :func:`octree_align` to take octrees as input for convenience.
    '''
    key = octree.key(depth, nempty)
    query = octree_query.key(depth, nempty)
    _, found = search_value(key, query)
    return found


def merge_octrees(octrees, depth=None):
    r''' Merges a list of octrees into one batch.
    
    Args:
      octrees (List[Octree]): A list of octrees to merge.
    '''
    
    # init and check
    octree = Octree(depth=octrees[0].depth, full_depth=octrees[0].full_depth,
                    batch_size=len(octrees), device=octrees[0].device)
    for i in range(1, octree.batch_size):
        condition = (octrees[i].depth == octree.depth and
                   octrees[i].full_depth == octree.full_depth and
                   octrees[i].device == octree.device)
        assert condition, 'The check of merge_octrees failed'
    
    # node num
    batch_nnum = th.stack(
        [octrees[i].nnum for i in range(octree.batch_size)], dim=1)
    batch_nnum_nempty = th.stack(
        [octrees[i].nnum_nempty for i in range(octree.batch_size)], dim=1)
    octree.nnum = th.sum(batch_nnum, dim=1)
    octree.nnum_nempty = th.sum(batch_nnum_nempty, dim=1)
    octree.batch_nnum = batch_nnum
    octree.batch_nnum_nempty = batch_nnum_nempty
    nnum_cum = cumsum(batch_nnum_nempty, dim=1, exclusive=True)
    if depth is None:
        depth = octree.depth
    
    # merge octre properties
    for d in range(depth+1):
        # key
        keys = [None] * octree.batch_size
        for i in range(octree.batch_size):
            key = octrees[i].keys[d] & ((1 << 48) - 1)  # clear the highest bits
            keys[i] = key | (i << 48)
        octree.keys[d] = th.cat(keys, dim=0)
        
        # children
        children = [None] * octree.batch_size
        for i in range(octree.batch_size):
            child = octrees[i].children[d].clone()  # !! `clone` is used here to avoid
            mask = child >= 0                       # !! modifying the original octrees
            child[mask] = child[mask] + nnum_cum[d, i]
            children[i] = child
        octree.children[d] = th.cat(children, dim=0)
        
        # features
        if octrees[0].features[d] is not None and d == octree.depth:
            features = [octrees[i].features[d] for i in range(octree.batch_size)]
            octree.features[d] = th.cat(features, dim=0)
        
        # normals
        if octrees[0].normals[d] is not None and d == octree.depth:
            normals = [octrees[i].normals[d] for i in range(octree.batch_size)]
            octree.normals[d] = th.cat(normals, dim=0)
        
        # points
        if octrees[0].points[d] is not None and d == octree.depth:
            points = [octrees[i].points[d] for i in range(octree.batch_size)]
            octree.points[d] = th.cat(points, dim=0)
    
    return octree


def octree_unnormalize_pts(pts, octree, depth, z_min, grid_size, grid_res, nempty=False):
    batch_nnum = th.bincount(octree.batch_id(depth, nempty=nempty), minlength=octree.batch_size)
    z_min = th.repeat_interleave(z_min, batch_nnum)
    pts[:, :2] = (grid_res * grid_size // 2) * pts[:, :2]
    pts[:, 2] = (0.5 * pts[:, 2] + 0.5) * grid_res * grid_size + z_min
    return pts
