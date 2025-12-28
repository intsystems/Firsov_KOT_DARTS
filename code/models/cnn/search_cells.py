""" CNN cell for architecture search """
import torch
import torch.nn as nn
import  models.cnn.ops as ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, primitives, drop=0.0, latency_manager=None):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.latency_manager = latency_manager

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i): 
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp(C, stride, primitives, drop)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        
        
        for edges, w_list in zip(self.dag, w_dag):
            if not isinstance(w_list, list) and len(w_list.shape) == 3:
                w_list = w_list.permute(1,2,0)
            if self.latency_manager is not None:
                for i, (s, _) in enumerate(zip(states, w_list)):
                    dummy_input = s.new_zeros((1, s.shape[1], s.shape[2], s.shape[3]))
                    mixed_op = edges[i]
                    for op in mixed_op._ops:
                        op_name = op.name if hasattr(op, 'name') else str(op.__class__.__name__).lower()
                        if 'zero' in op_name:
                            op_name = 'none'
                        try:
                            self.latency_manager.get_or_measure_latency(op_name, op, dummy_input)
                        except Exception:
                            pass
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        return s_out
