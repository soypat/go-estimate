package particle

import (
	filter "github.com/soypat/go-estimate"
	"gonum.org/v1/gonum/mat"
)

// Particle is Particle Filter
type Particle interface {
	// filter.Filter is dynamical system filter
	filter.Filter
	// Weights returns particle weights
	Weights() mat.Vector
}
