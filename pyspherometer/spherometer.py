import pathlib
import appdirs
import tomli_w
import tomli
import sympy as sy
import pandas as pd
import numpy as np
from rich.table import Table


class Spherometer:
    def __init__(
        self,
        *,
        a=None,
        b=None,
        c=None,
        h0=None,
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
        self.h0 = h0
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
        r_s, h, d_b = sy.symbols("r_s, h, d_b")
        return (r_s**2 - h**2) / (2 * h) - (d_b / 2)

    def expr_f(self) -> sy.Symbol:
        """Focal length of the mirror."""
        return self.expr_r_m() / 2

    def expr_f_ratio(self) -> sy.Symbol:
        """F-ratio of the mirror."""
        d = sy.symbols("d")
        return self.expr_f() / d

    def expr_df(self) -> sy.Symbol:
        """Uncertainty in the focal length of the mirror."""
        h, a, b, c, ds, dx, k, r_s = sy.symbols("h, a, b, c, Delta_s, Delta_x, k, r_s")

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
                (r_s / h) * (drsda + drsdb + drsdc) * dx
            - ((r_s**2 / (2 * h**2)) + h) * ds
            )
        )

    def func_f(self):
        a, b, c, d_b, h, r_s, k = sy.symbols("a, b, c, d_b, h, r_s, k")

        return sy.lambdify(
            [h],
            self.expr_f().subs(
                [(r_s, self.expr_r_s())]
            ).subs(
                [(k, self.expr_k())]
            ).subs(
                [(a, self.a), (b, self.b), (c, self.c), (d_b, self.d_b)]
            ),
        )

    def func_df(self):
        h, a, b, c, ds, dx, k, r_s, d_b = sy.symbols("h, a, b, c, Delta_s, Delta_x, k, r_s, d_b")

        return sy.lambdify(
            [h],
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
        a, b, c, h, d_b, d, r_s, k = sy.symbols("a, b, c, h, d_b, d, r_s, k")
        return sy.lambdify(
            [h, d],
            self.expr_f_ratio().subs(
                [(r_s, self.expr_r_s())]
            ).subs(
                [(k, self.expr_k())]
            ).subs(
                [(a, self.a), (b, self.b), (c, self.c), (d_b, self.d_b)]
            ),
        )

    def record(self, dial_meas=None, h_measured=None, file=None):
        if file is None:
            file = (
                pathlib.Path(appdirs.user_data_dir("megrez"))
                / "spherometer_records.csv"
            )
        file.parent.mkdir(parents=True, exist_ok=True)

        if dial_meas:
            h_measured = self.h0 - dial_meas

        data = {
            "datetime": [pd.Timestamp.today()],
            "h": [h_measured],
            "f": [self.func_f()(h_measured)],
            "df": [self.func_df()(h_measured)],
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

    def __str__(self):
        return vars(self)

    def records(self, file=None, aperture=None, pretty=True):
        if file is None:
            file = (
                pathlib.Path(appdirs.user_data_dir("megrez"))
                / "spherometer_records.csv"
            )
        df = pd.read_csv(file).set_index("datetime")

        if aperture is not None:
            df['F'] = df['f']/aperture

        if pretty:
            table = Table('Spherometer Records')
            for col in df.columns:
                table.add_column(col)

            for i, row in df.iterrows():
                table.add_row(str(i), *[str(val) for val in row.values])
            return table
        else:
            return df

    def expr_target_f_ratio(self):
        d_b, d, r_s, F = sy.symbols("d_b, d, r_s, F")

        # Reject negative root
        term = (4*F*d + d_b)
        return (-term + sy.sqrt(term**2 + 4*r_s**2))/2

    def func_target_f_ratio(self):
        a, b, c, k, F, r_s, d, d_b = sy.symbols("a, b, c, k, F, r_s, d, d_b")

        return sy.lambdify(
            [F, d],
            self.expr_target_f_ratio().subs(
                [(r_s, self.expr_r_s())]
            ).subs(
                [(k, self.expr_k())]
            ).subs(
                [(a, self.a), (b, self.b), (c, self.c), (d_b, self.d_b)]
            ),
        )
