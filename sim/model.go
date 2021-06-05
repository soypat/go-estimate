package sim

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// InitCond implements filter.InitCond
type InitCond struct {
	state *mat.VecDense
	cov   *mat.SymDense
}

// NewInitCond creates new InitCond and returns it
func NewInitCond(state mat.Vector, cov mat.Symmetric) *InitCond {
	s := &mat.VecDense{}
	s.CloneFromVec(state)

	c := mat.NewSymDense(cov.Symmetric(), nil)
	c.CopySym(cov)

	return &InitCond{
		state: s,
		cov:   c,
	}
}

// State returns initial state
func (c *InitCond) State() mat.Vector {
	state := mat.NewVecDense(c.state.Len(), nil)
	state.CloneFromVec(c.state)

	return state
}

// Cov returns initial covariance
func (c *InitCond) Cov() mat.Symmetric {
	cov := mat.NewSymDense(c.cov.Symmetric(), nil)
	cov.CopySym(c.cov)

	return cov
}

// Control system defines a linear model of a control system.
// It contains the System (A), input (B), Observation/Output (C)
// Feedthrough (D) and disturbance (E) matrices.
type ControlSystem struct {
	// A is internal state matrix
	A *mat.Dense
	// B is control matrix
	B *mat.Dense
	// C is output state matrix
	C *mat.Dense
	// D is output control matrix
	D *mat.Dense
	// E is Disturbance matrix
	E *mat.Dense
}

// Discrete is a basic model of a linear, discrete-time, dynamical system
type Discrete struct {
	ControlSystem
}

type Continuous struct {
	ControlSystem
}

// NewDiscrete creates a linear discrete-time model based on the control theory equations.
//
//  x[n+1] = A*x[n] + B*u[n] + E*z[n] (disturbances E not implemented yet)
//  y[n] = C*x[n] + D*u[n]
func NewDiscrete(A, B, C, D, E *mat.Dense) (*Discrete, error) {
	if A == nil {
		return nil, fmt.Errorf("system matrix must be defined for a model")
	}
	return &Discrete{ControlSystem: ControlSystem{A: A, B: B, C: C, D: D, E: E}}, nil
}

// NewContinous creates a linear continuous-time model based on the control theory equations
// which is advanced by timestep dt.
//
//  dx/dt = A*x + B*u + E*z (disturbances E not implemented yet)
//  y = C*x + D*u
func NewContinous(A, B, C, D, E *mat.Dense) (*Continuous, error) {
	if A == nil {
		return nil, fmt.Errorf("system matrix must be defined for a model")
	}
	return &Continuous{ControlSystem: ControlSystem{A: A, B: B, C: C, D: D, E: E}}, nil
}

// ToDiscrete creates a discrete-time model from a continuous time model
// using Ts as the sampling time.
//
// It is calculated using Euler's method, an approximation valid for small timesteps.
func (c *Continuous) ToDiscrete(Ts float64) (*Discrete, error) {
	var A, B, C, D, E *mat.Dense
	if c.A == nil {
		return nil, fmt.Errorf("system matrix must be defined for a model")
	}
	nx, _, _, _ := c.SystemDims()
	A = mat.DenseCopyOf(c.A)

	if c.B != nil {
		B = mat.DenseCopyOf(c.B)
		B.Scale(Ts, B)
	}
	if c.E != nil {
		E = mat.DenseCopyOf(c.E)
		E.Scale(Ts, E)
	}

	if c.C != nil {
		C = mat.DenseCopyOf(c.C)
	}
	if c.D != nil {
		D = mat.DenseCopyOf(c.D)
	}
	// x[n+1] = (I + A*dt)x[n] + dt*B*u[n]
	A.Scale(Ts, A)
	for i := 0; i < nx; i++ {
		A.Set(i, i, A.At(i, i)+1.)
	}
	return &Discrete{ControlSystem{A: A, B: B, C: C, D: D, E: E}}, nil
}

// Propagate propagates returns the next internal state x
// of a linear, continuous-time system given an input vector u and a
// disturbance input z. (wd is process noise, z not implemented yet)
func (ct *Discrete) Propagate(x, u, wd mat.Vector) (mat.Vector, error) {
	nx, nu, _, _ := ct.SystemDims()
	if u != nil && u.Len() != nu {
		return nil, fmt.Errorf("invalid input vector")
	}

	if x.Len() != nx {
		return nil, fmt.Errorf("invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(ct.A, x)
	if u != nil && ct.B != nil {
		outU := new(mat.Dense)
		outU.Mul(ct.B, u)

		out.Add(out, outU)
	}

	if wd != nil && wd.Len() == nx {
		out.Add(out, wd)
	}
	return out.ColView(0), nil
}

// Propagate propagates returns the next internal state x
// of a linear, continuous-time system given an input vector u and a
// disturbance input z. (wd is process noise, z not implemented yet). It propagates
// the solution by a timestep `dt`.
func (ct *Continuous) Propagate(x, u, wd mat.Vector, dt float64) (mat.Vector, error) {
	nx, nu, _, _ := ct.SystemDims()
	if u != nil && u.Len() != nu {
		return nil, fmt.Errorf("invalid input vector")
	}

	if x.Len() != nx {
		return nil, fmt.Errorf("invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(ct.A, x)
	if u != nil && ct.B != nil {
		outU := new(mat.Dense)
		outU.Mul(ct.B, u)

		out.Add(out, outU)
	}

	if wd != nil && wd.Len() == nx { // TODO change _nx to _nz when switching to z and disturbance matrix implementation
		// outZ := new(mat.Dense) // TODO add E disturbance matrix
		// outZ.Mul(b.E, z)
		// out.Add(out, outZ)
		out.Add(out, wd)
	}
	// integrate the first order derivatives calculated: dx/dt = A*x + B*u + wd
	out.Scale(dt, out)
	out.Add(x, out)
	return out.ColView(0), nil
}

// Observe returns external/observable state given internal state x and input u.
// wn is added to the output as a noise vector.
func (b *ControlSystem) Observe(x, u, wn mat.Vector) (mat.Vector, error) {
	nx, nu, ny, _ := b.SystemDims()
	if u != nil && u.Len() != nu {
		return nil, fmt.Errorf("invalid input vector")
	}

	if x.Len() != nx {
		return nil, fmt.Errorf("invalid state vector")
	}

	out := new(mat.Dense)
	out.Mul(b.C, x)

	if u != nil && b.D != nil {
		outU := new(mat.Dense)
		outU.Mul(b.D, u)

		out.Add(out, outU)
	}

	if wn != nil && wn.Len() == ny {
		out.Add(out, wn)
	}

	return out.ColView(0), nil
}

// SystemDims returns internal state length (nx), input vector length (nu),
// external/observable/output state length (ny) and disturbance vector length (nz).
func (b *ControlSystem) SystemDims() (nx, nu, ny, nz int) {
	nx, _ = b.A.Dims()
	if b.B != nil {
		_, nu = b.B.Dims()
	}
	ny, _ = b.C.Dims()
	if b.E != nil {
		_, nz = b.E.Dims()
	}
	return nx, nu, ny, nz
}

// SystemMatrix returns state propagation matrix `A`.
func (b *ControlSystem) SystemMatrix() mat.Matrix {
	m := &mat.Dense{}
	m.CloneFrom(b.A)

	return m
}

// ControlMatrix returns state propagation control matrix `B`.
func (b *ControlSystem) ControlMatrix() mat.Matrix {
	m := &mat.Dense{}
	if b.B != nil {
		m.CloneFrom(b.B)
	}

	return m
}

// OutputMatrix returns observation matrix `C`.
func (b *ControlSystem) OutputMatrix() mat.Matrix {
	m := &mat.Dense{}
	m.CloneFrom(b.C)

	return m
}

// FeedForwardMatrix returns observation control matrix `D`.
func (b *ControlSystem) FeedForwardMatrix() mat.Matrix {
	m := &mat.Dense{}
	if b.D != nil {
		m.CloneFrom(b.D)
	}
	return m
}
