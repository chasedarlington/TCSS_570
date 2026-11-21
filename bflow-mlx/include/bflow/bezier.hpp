#pragma once

#include <mlx/mlx.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace bflow {

namespace mx = mlx::core;

// Pixel-space (x, y) coordinates with a broadcastable batch dimension.
inline mx::array coordinate_grid(int height, int width) {
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("coordinate grid dimensions must be positive");
    }
    const auto x = mx::arange(static_cast<double>(width), mx::float32);
    const auto y = mx::arange(static_cast<double>(height), mx::float32);
    const mx::Shape extent{1, height, width};
    return mx::stack({
        mx::broadcast_to(mx::reshape(x, {1, 1, width}), extent),
        mx::broadcast_to(mx::reshape(y, {1, height, 1}), extent),
    }, -1);
}

class BezierBasis {
public:
    BezierBasis(int degree, std::span<const double> timestamps)
        : degree_(degree), coefficients_(build_basis(degree, timestamps)) {}

    // Controls [B,H,W,2*degree] follow the checkpoint's component-major
    // channel order [P1.x,...,Pn.x,P1.y,...,Pn.y]. P0 is fixed at zero.
    // The output [K,B,H,W,2] is displacement at each requested timestamp.
    [[nodiscard]] mx::array evaluate(const mx::array& control_points) const {
        if (control_points.ndim() != 4 || control_points.shape(3) != 2 * degree_) {
            throw std::invalid_argument("Bezier controls must be [B,H,W,2*degree]");
        }
        const auto controls = mx::reshape(control_points, {
            1, control_points.shape(0), control_points.shape(1),
            control_points.shape(2), 2, degree_,
        });
        return mx::sum(mx::multiply(controls, coefficients_), -1);
    }

private:
    static mx::array build_basis(int degree, std::span<const double> timestamps) {
        if (degree < 1 || degree > std::numeric_limits<int>::max() / 2
            || timestamps.empty()
            || timestamps.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("Bezier basis needs a positive degree and timestamps");
        }

        std::vector<float> coefficients;
        coefficients.reserve(timestamps.size() * static_cast<std::size_t>(degree));
        for (const double t : timestamps) {
            if (!std::isfinite(t) || t < 0.0 || t > 1.0) {
                throw std::invalid_argument("Bezier timestamps must lie in [0,1]");
            }
            double binomial = 1.0;
            for (int i = 1; i <= degree; ++i) {
                binomial *= static_cast<double>(degree - i + 1) / i;
                const double weight = binomial * std::pow(1.0 - t, degree - i)
                                    * std::pow(t, i);
                coefficients.push_back(static_cast<float>(weight));
            }
        }
        return mx::array(coefficients.begin(), {
            static_cast<int>(timestamps.size()), 1, 1, 1, 1, degree,
        });
    }

    int degree_;
    mx::array coefficients_;
};

}  // namespace bflow
