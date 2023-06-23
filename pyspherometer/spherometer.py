import pathlib
import appdirs
import tomli_w
import tomli
import sympy as sy
import pandas as pd
import numpy as np


class Spherometer:
    def __init__(
        self,
        *,
        a=None,
        b=None,
        c=None,
        s0=None,
        d_b=None,
        ds=None,
        dx=None,
        a_meas=None,
        b_meas=None,
        c_meas=None,
        **kwargs,
    ):
        if a_meas and b_meas and c_meas:
            self.a = a_meas - d_b
            self.b = b_meas - d_b
            self.c = c_meas - d_b
        else:
            self.a = a
            self.b = b
            self.c = c
        self.s0 = s0
        self.d_b = d_b
        self.ds = ds
        self.dx = dx

    def expr_k(self) -> sy.Symbol:
        """K-value."""
        a, b, c = sy.symbols("a, b, c")
        return (a + b + c) / 2

    def expr_r_s(self) -> sy.Symbol:
        """Radius of the spherometer."""
        a, b, c, k = sy.symbols("a, b, c, k")
        return (a * b * c) / (4 * sy.sqrt((k * (k - a) * (k - b) * (k - c))))

    def expr_r_m(self) -> sy.Symbol:
        """Radius of the mirror."""
        r_s, s, d_b = sy.symbols("r_s, s, d_b")
        return (r_s**2 - s**2) / (2 * s) - (d_b / 2)

    def expr_f(self) -> sy.Symbol:
        """Focal length of the mirror."""
        return self.expr_r_m() / 2

    def expr_f_ratio(self) -> sy.Symbol:
        """F-ratio of the mirror."""
        d = sy.symbols("d")
        return self.expr_f() / d

    def expr_df(self) -> sy.Symbol:
        """Uncertainty in the focal length of the mirror."""
        s, a, b, c, ds, dx, k, r_s = sy.symbols("s, a, b, c, Delta_s, Delta_x, k, r_s")

        # Denominator is shared by all 3 terms
        q = (-(k**4) + (a * b + a * c + b * c) * k**2 - k * a * b * c) ** (3 / 2)

        drsda = r_s / a - (a * b * c / 8) * (
            (-2 * k**3 + (b + c) * k**2 + (a * b + a * c) * k - a * b * c / 2) / q
        )
        drsdb = r_s / b - (a * b * c / 8) * (
            (-2 * k**3 + (a + c) * k**2 + (b * a + b * c) * k - a * b * c / 2) / q
        )
        drsdc = r_s / c - (a * b * c / 8) * (
            (-2 * k**3 + (a + b) * k**2 + (c * b + c * a) * k - a * b * c / 2) / q
        )

        return np.abs(
            (1 / 2) * (
                (r_s / s) * (drsda + drsdb + drsdc) * dx
            - ((r_s**2 / (2 * s**2)) + s) * ds
            )
        )

    def func_f(self):
        a, b, c, d_b, s, r_s, k = sy.symbols("a, b, c, d_b, s, r_s, k")

        return sy.lambdify(
            [s],
            self.expr_f().subs(
                [(r_s, self.expr_r_s())]
            ).subs(
                [(k, self.expr_k())]
            ).subs(
                [(a, self.a), (b, self.b), (c, self.c), (d_b, self.d_b)]
            ),
        )

    def func_df(self):
        s, a, b, c, ds, dx, k, r_s, d_b = sy.symbols("s, a, b, c, Delta_s, Delta_x, k, r_s, d_b")

        return sy.lambdify(
            [s],
            self.expr_df().subs(
                [(r_s, self.expr_r_s())]
            ).subs(
                [(k, self.expr_k())]
            ).subs(
                [
                    (a, self.a),
                    (b, self.b),
                    (c, self.c),
                    (d_b, self.d_b),
                    (ds, self.ds),
                    (dx, self.dx),
                ]
            ),
        )


    def func_f_ratio(self):
        a, b, c, s, d_b, d, r_s, k = sy.symbols("a, b, c, s, d_b, d, r_s, k")
        return sy.lambdify(
            [s, d],
            self.expr_f_ratio().subs(
                [(r_s, self.expr_r_s())]
            ).subs(
                [(k, self.expr_k())]
            ).subs(
                [(a, self.a), (b, self.b), (c, self.c), (d_b, self.d_b)]
            ),
        )

    def record(self, dial_meas=None, sagitta=None, file=None):
        if file is None:
            file = (
                pathlib.Path(appdirs.user_data_dir("megrez"))
                / "spherometer_records.csv"
            )
        file.parent.mkdir(parents=True, exist_ok=True)

        a, b, c, s, d_b, ds, dx = sy.symbols("a, b, c, s, d_b, Delta_s, Delta_x")

        if dial_meas:
            sagitta = self.s0 - dial_meas

        data = {
            "datetime": [pd.Timestamp.today()],
            "s": [sagitta],
            "f": [self.func_f()(sagitta)],
            "df": [self.func_df()(sagitta)],
        }

        if pathlib.Path(file).exists():
            df = pd.concat([pd.read_csv(file), pd.DataFrame(data)]).set_index(
                "datetime"
            )
        else:
            df = pd.DataFrame(data).set_index("datetime")

        df.to_csv(file)

    def save(self, file=None):
        if file is None:
            path = pathlib.Path(appdirs.user_config_dir("megrez")) / "spherometer.toml"
        else:
            path = pathlib.Path(file)

        if path.exists():
            raise ValueError(
                f"Path to {file} already exists! Delete the file if you want to erase "
                "your config."
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            tomli_w.dump(vars(self), f)

    @classmethod
    def load(cls, file=None):
        if file is None:
            path = pathlib.Path(appdirs.user_config_dir("megrez")) / "spherometer.toml"
        else:
            path = pathlib.Path(file)

        with open(path, "rb") as f:
            config = tomli.load(f)

        return cls(**config)
