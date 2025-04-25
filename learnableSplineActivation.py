import torch
import torch.nn as nn

class LearnableBSplineActivation(nn.Module):
    def __init__(self, num_control_points=5, degree=3, init_scale=1.0):

        super(LearnableBSplineActivation, self).__init__()
        self.num_control_points = num_control_points
        self.degree = degree
        
        # Learnable control points
        self.control_points = nn.Parameter(torch.randn(num_control_points) * init_scale)
        
        # knot vector
        self.register_buffer('knots', torch.linspace(-2, 2, num_control_points + degree + 1))
        
        # Cache for basis functions
        self.register_buffer('basis_cache', torch.zeros(1))
        self.cache_size = 0
        
    def basis_function(self, x, i, k):
        if k == 0:
            return torch.where((self.knots[i] <= x) & (x < self.knots[i+1]), 
                             torch.ones_like(x), torch.zeros_like(x))
        
        denom1 = self.knots[i+k] - self.knots[i]
        term1 = torch.zeros_like(x) if denom1 == 0 else (x - self.knots[i]) / denom1 * self.basis_function(x, i, k-1)
        
        denom2 = self.knots[i+k+1] - self.knots[i+1]
        term2 = torch.zeros_like(x) if denom2 == 0 else (self.knots[i+k+1] - x) / denom2 * self.basis_function(x, i+1, k-1)
        
        return term1 + term2
    
    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(self.num_control_points):
            basis = self.basis_function(x, i, self.degree)
            output += self.control_points[i] * basis
            
        return output