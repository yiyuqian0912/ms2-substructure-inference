"""Verify fragment masses without treating a merged readout bin as exact truth."""
import torch
from .data import ELEMENTS, tolerance


def node_mass_bounds(record):
    if 'node_masses' in record:
        mass=record['node_masses'].detach().cpu().double()
        return mass, torch.zeros_like(mass), 'exported_node_masses'
    # Older caches retain float32 mass/precursor ratios. Use a conservative
    # rounding interval, rather than labelling a reconstructed value exact.
    ratio=record['node_num'][:,len(ELEMENTS)].detach().cpu().float()
    lower=torch.nextafter(ratio,torch.full_like(ratio,-torch.inf))
    upper=torch.nextafter(ratio,torch.full_like(ratio,torch.inf))
    precursor=float(record['precursor_mz'])
    mass=ratio.double()*precursor
    radius=torch.maximum(ratio.double()-lower.double(),upper.double()-ratio.double())*.5*precursor
    radius=radius+mass.abs()*torch.finfo(torch.float64).eps*8
    return mass,radius,'cached_ratio_with_rounding_bound'


def check_peak(masses, bounds, mz, cfg):
    error=(masses-float(mz)).abs()
    limit=tolerance(float(mz),cfg)
    compatible=error+bounds<=limit
    incompatible=error-bounds>limit
    status=['compatible' if bool(a) else 'incompatible' if bool(b) else 'boundary_uncertain'
            for a,b in zip(compatible,incompatible)]
    return [dict(node_mz_estimate=float(m),mass_error_da=float(e),
                 mass_error_bound_da=float(r),mass_tolerance_da=limit,mass_compatibility=s)
            for m,e,r,s in zip(masses,error,bounds,status)]
