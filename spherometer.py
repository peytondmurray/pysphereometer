import pathlib
import appdirs
import tomli_w
import tomli
import sympy as sy
import pandas as pd


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
        a, b, c = sy.symbols("a, b, c")
        k = self.expr_k()
        return (a * b * c) / (4 * sy.sqrt((k * (k - a) * (k - b) * (k - c))))

    def expr_r_m(self) -> sy.Symbol:
        """Radius of the mirror."""
        s, d_b = sy.symbols("s, d_b")
        return (self.expr_r_s() ** 2 - s**2) / (2 * s) - (d_b / 2)

    def expr_f(self) -> sy.Symbol:
        """Focal length of the mirror."""
        return self.expr_r_m() / 2

    def expr_f_ratio(self) -> sy.Symbol:
        """F-ratio of the mirror."""
        d = sy.symbols("d")
        return self.expr_f() / d

    def expr_df(self) -> sy.Symbol:
        """Uncertainty in the focal length of the mirror."""
        s, a, b, c, ds, dx = sy.symbols("s, a, b, c, Delta_s, Delta_x")
        k = self.expr_k()
        r_s = self.expr_r_s()

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

        return (1 / 2) * (
            (r_s / s) * (drsda + drsdb + drsdc) * dx
            - ((r_s**2 / (2 * s**2)) + s) * ds
        )

    def func_f(self):
        a, b, c, d_b, s = sy.symbols("a, b, c, d_b, s")

        return sy.lambdify(
            [s],
            self.expr_f().subs(
                [(a, self.a), (b, self.b), (c, self.c), (d_b, self.d_b)]
            ),
        )

    def func_f_ratio(self):
        a, b, c, s, d_b, d = sy.symbols("a, b, c, s, d_b, d")
        return sy.lambdify(
            [s, d],
            self.expr_f_ratio().subs(
                [(a, self.a), (b, self.b), (c, self.c), (d_b, self.d_b)]
            ),
        )

    def record(self, dial_meas=None, sagitta=None, file=None):
        if file is None:
            file = (
                pathlib.Path(appdirs.user_data_dir("megrez"))
                / "spherometer_records.csv"
            )

        a, b, c, s, d_b, ds, dx = sy.symbols("a, b, c, s, d_b, Delta_s, Delta_x")

        if dial_meas:
            sagitta = self.s0 - dial_meas

        data = {
            "datetime": [pd.Timestamp.today()],
            "s": [sagitta],
            "f": [
                sy.lambdify([a, b, c, d_b, s], self.expr_f())(
                    self.a,
                    self.b,
                    self.c,
                    self.d_b,
                    sagitta,
                )
            ],
            "df": [
                sy.lambdify([a, b, c, d_b, s, ds, dx], self.expr_df())(
                    self.a, self.b, self.c, self.d_b, sagitta, self.ds, self.dx
                )
            ],
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
            path = pathlib.Path(appdirs.user_config_dir("megrez")) / "megrez.toml"
        else:
            path = pathlib.Path(file)

        if path.exists():
            raise ValueError(
                f"Path to {file} already exists! Delete the file if you want to erase your config."
            )

        with open(path, "wb") as f:
            tomli_w.dump(vars(self), f)

    @classmethod
    def load(cls, file=None):
        if file is None:
            path = pathlib.Path(appdirs.user_config_dir("megrez")) / "megrez.toml"
        else:
            path = pathlib.Path(file)

        with open(path, "rb") as f:
            config = tomli.load(f)

        return cls(**config)
