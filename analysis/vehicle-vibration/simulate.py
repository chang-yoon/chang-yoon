"""Portfolio extension, 2026-09-13. Not the original MATLAB submission.

Run: python simulate.py
Dependency: numpy. Outputs are written beside this file.
Exact constant-input propagation; damper impulses at ideal road steps are included.
"""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parent
M, K0, C0 = 45.0, 218000.0, 1000.0


def response(k, c, speed, dt=0.001):
    # Explicit interpretation: 10 m plateaus, +/-5 mm, road ends at 60 m.
    # Begin at static equilibrium on the first plateau; observe 2 s after exit.
    interval = 10 / (speed / 3.6)
    levels = [-.005, .005, -.005, .005, -.005, .005, 0.0]
    durations = [interval] * 6 + [2.0]
    A = np.array([[0., 1.], [-k/M, -c/M]])
    eig, vectors = np.linalg.eig(A)
    inv = np.linalg.inv(vectors)
    x, v, previous, offset = levels[0], 0., levels[0], 0.
    times, roads, positions = [], [], []
    for y, duration in zip(levels, durations):
        # Integrating m*x'' + c*x' + k*x = c*y' + k*y over a jump:
        v += c/M * (y - previous)
        initial = np.array([x-y, v])
        t = np.linspace(0., duration, int(np.ceil(duration/dt))+1)
        state = np.real(vectors @ ((inv @ initial)[:, None] * np.exp(eig[:, None]*t)))
        times.extend(offset+t)
        roads.extend(np.full_like(t, y))
        positions.extend(state[0]+y)
        x, v = state[:, -1] + np.array([y, 0.])
        previous, offset = y, offset+duration
    return np.array(times), np.array(roads), np.array(positions)


def peak(k, c, speed, dt=.001):
    return float(np.max(np.abs(response(k, c, speed, dt)[2])))


def main():
    # Search declared equivalent stiffness domain. This is NOT a feasible
    # suspension-only redesign with fixed kt=200000 and k=ks+kt.
    candidates = [(float(k), float(c)) for k in np.geomspace(1000, K0, 60)
                  for c in [950., 1000., 1050.]]
    baseline = peak(K0, C0, 80)
    ranked = [(peak(k, c, 80), k, c) for k, c in candidates]
    best = min(ranked)
    # Choose largest sampled stiffness achieving 20%, limiting parameter change.
    feasible = [r for r in ranked if r[0] <= .8*baseline]
    chosen = max(feasible, key=lambda r: (r[1], -abs(r[2]-C0))) if feasible else best
    _, k, c = chosen
    rows = []
    for speed in [80, 20]:
        t, road, old = response(K0, C0, speed, .0005)
        _, _, new = response(k, c, speed, .0005)
        p0, p1 = float(max(abs(old))), float(max(abs(new)))
        fine = peak(k, c, speed, .00025)
        assert abs(fine-p1) < 1e-5, 'Peak resolution check failed'
        rows.append(dict(speed_kmh=speed, baseline_peak_mm=p0*1000,
                         revised_peak_mm=p1*1000, reduction_percent=100*(1-p1/p0),
                         refinement_difference_mm=abs(fine-p1)*1000))
        # Compact trace for inspection; peaks above use the full resolution.
        np.savetxt(OUT/f'trace-{speed}kmh.csv',
                   np.column_stack([t,road*1000,old*1000,new*1000])[::10],
                   delimiter=',', fmt='%.8f',
                   header='time_s,road_mm,baseline_mm,revised_mm',comments='')
    # Independent RK4 check of one segment propagation.
    A = np.array([[0., 1.], [-k/M, -c/M]])
    eig, V = np.linalg.eig(A)
    a = np.real(V @ np.diag(np.exp(eig*.123)) @ np.linalg.inv(V))
    check = np.eye(2)
    h = .123 / 12300
    for _ in range(12300):
        r1 = A @ check
        r2 = A @ (check+h*r1/2)
        r3 = A @ (check+h*r2/2)
        r4 = A @ (check+h*r3)
        check += h*(r1+2*r2+2*r3+r4)/6
    assert np.allclose(a, check, atol=1e-8)
    # Feasible stiffness with fixed tire spring and nonnegative suspension spring.
    bounded = min((peak(k2,c2,80),float(k2),c2)
                  for k2 in np.linspace(200000,K0,20) for c2 in [950.,1000.,1050.])
    data = dict(analysis_date='2026-09-13', baseline=dict(m_kg=M,k_N_m=K0,c_Ns_m=C0),
                revised=dict(k_N_m=k,c_Ns_m=c), candidate_count=len(candidates),
                selection='Largest sampled stiffness meeting 20 percent at 80 km/h',
                road='10 m alternating plateaus +/-5 mm; 60 m road then zero for 2 s',
                initial_condition='Static equilibrium at -5 mm; velocity zero',
                results=rows,
                fixed_tire_search=dict(k_N_m=bounded[1], c_Ns_m=bounded[2],
                                       reduction_percent=100*(1-bounded[0]/baseline)))
    (OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data,indent=2))


if __name__ == '__main__':
    main()
