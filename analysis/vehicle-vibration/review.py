"""Reproduce coursework FRFs and test sensitivity to road-transition assumptions.

New portfolio review, not an original submission. Run simulate.py first.
Dependencies are in requirements.txt. All paths are relative to this file.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulate import M, K0, C0, response

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'


def quarter_car(f):
    mass = np.diag([300., 45.])
    damping = np.array([[1000., -1000.], [-1000., 1000.]])
    stiffness = np.array([[18000., -18000.], [-18000., 218000.]])
    w = 2*np.pi*f
    impedance = (stiffness[None, :, :] - w[:, None, None]**2*mass
                 + 1j*w[:, None, None]*damping)
    forcing = np.broadcast_to([0., 200000.], (len(f), 2))
    h = np.linalg.solve(impedance, forcing[..., None])[..., 0]
    # Independent closed-form inverse verifies ordering, signs and normalization.
    det = impedance[:, 0, 0]*impedance[:, 1, 1]-impedance[:, 0, 1]**2
    explicit = np.column_stack([-200000*impedance[:, 0, 1]/det,
                                 200000*impedance[:, 0, 0]/det])
    assert np.allclose(h, explicit, atol=1e-11)
    assert np.allclose(h[0], [1, 1]), 'Static unit-road response'
    eigenvalues = np.linalg.eigvals(np.linalg.solve(mass, stiffness))
    assert np.max(abs(eigenvalues.imag)) < 1e-8
    natural = np.sqrt(np.sort(eigenvalues.real))/(2*np.pi)
    return h, natural


def relative_frf(f, k=K0, c=C0):
    w = 2*np.pi*f
    return M*w*w/(k-M*w*w+1j*c*w)


def linear_input_response(t, y, k, c, initial=None):
    # p = xdot - c*y/m removes the need to differentiate the road numerically.
    # sdot = A*s + B*y; exact propagation of the linearly interpolated input.
    h = t[1]-t[0]
    a = np.array([[0., 1.], [-k/M, -c/M]])
    b = np.array([c/M, k/M-c*c/(M*M)])
    lam, v = np.linalg.eig(a)
    phi = np.real(v @ np.diag(np.exp(lam*h)) @ np.linalg.inv(v))
    g0 = np.linalg.solve(a, (phi-np.eye(2)) @ b)
    g1 = np.linalg.solve(a, np.linalg.solve(a, (phi-np.eye(2)-a*h) @ b))/h
    state = np.array([y[0], -c*y[0]/M]) if initial is None else np.array(initial)
    xs = np.empty(len(t))
    xs[0] = state[0]
    for i in range(len(t)-1):
        state = phi @ state + g0*y[i] + g1*(y[i+1]-y[i])
        xs[i+1] = state[0]
    return xs


def smooth_road(speed, width, dt):
    velocity = speed/3.6
    duration = 60/velocity + 2
    t = np.linspace(0, duration, int(np.ceil(duration/dt))+1)
    distance = t*velocity
    y = np.full_like(t, -.005)
    # Changes begin at the same positions as the ideal step case.
    for position, jump in [(10,.01),(20,-.01),(30,.01),(40,-.01),(50,.01),(60,-.005)]:
        s = np.clip((distance-position)/width, 0, 1)
        y += jump*(1-np.cos(np.pi*s))/2
    return t, y


def style(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)


def save(fig, name):
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    data = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    k, c = data['revised']['k_N_m'], data['revised']['c_Ns_m']
    f = np.linspace(0, 20, 20001)
    h, natural = quarter_car(f)
    sprung_peak = float(f[np.argmax(abs(h[:, 0]))])
    wheel_peak = float(f[np.argmax(abs(h[:, 1]))])
    rel = relative_frf(f)
    rel_peak = int(np.argmax(abs(rel)))
    fig, axes = plt.subplots(2,1,figsize=(9,7),layout='constrained')
    axes[0].plot(f, abs(h[:,0]), label=f'Sprung mass (peak {sprung_peak:.3f} Hz)')
    axes[0].plot(f, abs(h[:,1]), label=f'Unsprung mass (peak {wheel_peak:.3f} Hz)')
    style(axes[0], 'Frequency (Hz)', 'Displacement ratio |X/Y|', '2-DOF quarter-car: reproduced frequency response')
    axes[1].plot(f, .01*abs(rel)*1000, label='1-DOF relative displacement, road amplitude 10 mm')
    axes[1].scatter(f[rel_peak], .01*abs(rel[rel_peak])*1000, color='black', s=20)
    style(axes[1], 'Frequency (Hz)', 'Relative amplitude (mm)', 'General 1-DOF model: a separate base-excitation assumption')
    save(fig, 'frequency-response.png')
    # Same OUTPUT (wheel absolute displacement) and INPUT (road displacement).
    # This comparison must not mix relative displacement with absolute displacement.
    w = 2*np.pi*f
    denominator = K0-M*w*w+1j*C0*w
    fixed_body = 200000/denominator
    general_base = (K0+1j*C0*w)/denominator
    coupling = 18000+1j*C0*w
    condensed = 200000/(218000-M*w*w+1j*C0*w-coupling**2/(18000-300*w*w+1j*C0*w))
    assert np.allclose(condensed,h[:,1],atol=1e-10)
    assert abs(fixed_body[0]-200000/K0) < 1e-12
    assert abs(general_base[0]-1) < 1e-12
    fig, ax = plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(f,abs(h[:,1]),label='Original 2-DOF: wheel / road')
    ax.plot(f,abs(fixed_body),label='1-DOF: sprung mass fixed')
    ax.plot(f,abs(general_base),label='1-DOF: common moving base',linestyle='--')
    style(ax,'Frequency (Hz)','Absolute displacement ratio |X/Y|',
          'Different boundary conditions give different transfer functions')
    save(fig, 'model-comparison.png')
    fig, axes = plt.subplots(2,1,figsize=(9,7),layout='constrained')
    for factor in [.5,1,1.5]:
        axes[0].plot(f, abs(relative_frf(f,K0*factor,C0))*10,label=f'k = {factor:g} x baseline')
    for factor in [.5,1,2]:
        axes[1].plot(f, abs(relative_frf(f,K0,C0*factor))*10,label=f'c = {factor:g} x baseline')
    style(axes[0], 'Frequency (Hz)', 'Relative amplitude (mm)', 'Stiffness comparison (damping fixed)')
    style(axes[1], 'Frequency (Hz)', 'Relative amplitude (mm)', 'Damping comparison (stiffness fixed)')
    save(fig, 'parameter-comparison.png')
    fig, axes = plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for ax, speed in zip(axes,[80,20]):
        t, y, old = response(K0,C0,speed,.0005)
        _, _, new = response(k,c,speed,.0005)
        ax.plot(t,y*1000,color='gray',lw=.8,label='Ideal road')
        ax.plot(t,old*1000,label='Baseline')
        ax.plot(t,new*1000,label='Revised equivalent model',lw=1)
        style(ax,'Time (s)','Absolute displacement (mm)',f'{speed} km/h: ideal step-road response')
    save(fig, 'time-response.png')

    # Solver validation: constant equilibrium and independent harmonic solution.
    tcheck = np.linspace(0,5,10001)
    const = linear_input_response(tcheck,np.full_like(tcheck,.005),k,c)
    assert np.max(abs(const-.005)) < 1e-10
    w = 2*np.pi*3
    ycheck = .01*np.sin(w*tcheck)
    numerical = linear_input_response(tcheck,ycheck,k,c,initial=[0.,0.])
    transfer = (k+1j*c*w)/(k-M*w*w+1j*c*w)
    theory = np.imag(.01*transfer*np.exp(1j*w*tcheck))
    harmonic_error = float(max(abs(numerical[tcheck>4]-theory[tcheck>4])))
    assert harmonic_error < 2e-7
    rows = []
    fig, axes = plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for ax, speed in zip(axes,[80,20]):
        for width in [.1,.5,1.]:
            t,y = smooth_road(speed,width,.0005)
            old = linear_input_response(t,y,K0,C0)
            new = linear_input_response(t,y,k,c)
            tf,yf = smooth_road(speed,width,.00025)
            oldf = linear_input_response(tf,yf,K0,C0)
            newf = linear_input_response(tf,yf,k,c)
            p0,p1 = float(max(abs(old))),float(max(abs(new)))
            error = float(max(abs(max(abs(oldf))-p0),abs(max(abs(newf))-p1)))
            assert error < 2e-6, 'Smooth-input peak refinement check'
            rows.append(dict(speed_kmh=speed,transition_width_m=width,
                             baseline_peak_mm=p0*1000,revised_peak_mm=p1*1000,
                             reduction_percent=100*(1-p1/p0), refinement_error_mm=error*1000))
            if width == .5:
                # First rise, aligned to show the different transition duration.
                tau=t-10/(speed/3.6)
                use=(tau>=-.05)&(tau<=.4)
                ax.plot(tau[use],y[use]*1000,color='gray',label='0.5 m smooth road')
                ax.plot(tau[use],old[use]*1000,label='Baseline')
                ax.plot(tau[use],new[use]*1000,label='Revised equivalent model')
                style(ax,'Time from first rise (s)','Absolute displacement (mm)',f'{speed} km/h: sensitivity to finite road transition')
    save(fig, 'road-sensitivity.png')
    result = dict(review_date='2026-09-13',natural_frequencies_hz=natural.tolist(),
                  reproduced_peaks_hz=dict(sprung=sprung_peak,unsprung=wheel_peak,
                                           relative=float(f[rel_peak])),
                  relative_peak_mm=float(abs(rel[rel_peak])*10),
                  harmonic_check_max_error_mm=harmonic_error*1000,
                  model_dc_gains=dict(original_2dof=float(abs(h[0,1])),
                                      fixed_body=float(abs(fixed_body[0])),
                                      common_base=float(abs(general_base[0]))),
                  smooth_road_results=rows,
                  checks=['FRF direct solve versus explicit inverse','Static FRF limit',
                          'Exact frequency condensation versus original wheel response',
                          'Constant-input equilibrium','Harmonic steady-state analytical solution',
                          '0.5 ms versus 0.25 ms peak convergence'])
    (OUT/'review-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
