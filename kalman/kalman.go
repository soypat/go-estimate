package kalman

import (
	filter "github.com/soypat/go-estimate"
	"gonum.org/v1/gonum/mat"
)

// Kalman is Kalman Filter
type Kalman interface {
	// filter.Filter is dynamical system filter
	filter.Filter
	// Cov returns Kalman filter state covariance
	Cov() mat.Symmetric
	// Gain returns Kalman filter gain
	Gain() mat.Matrix
}
