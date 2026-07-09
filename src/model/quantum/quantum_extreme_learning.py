import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pennylane as qml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_smoted_raw_train_4.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "pre-processed" / "loan_data_raw_test_1.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "pre-processed"


class QELMIsing:
    def __init__(self):
        self.train_df = pd.read_csv(TRAIN_DATA_PATH)
        self.test_df = pd.read_csv(TEST_DATA_PATH)

        # Sensitive attributes: EXCLUDED from the model, kept only for fairness auditing
        # self.sensitive_cols = ['person_gender', 'person_age']

        self.target = 'loan_status'

        # Feature groups
        self.ordinal_cols = ['person_education']
        self.nominal_cols = ['person_gender', 'person_home_ownership', 'loan_intent']
        self.numerical_cols = ['person_age', 'person_income', 'loan_int_rate', 'cb_person_cred_hist_length', 'person_emp_exp',
                               'loan_amnt', 'loan_percent_income', 'credit_score']

        self.feature_cols = self.ordinal_cols + self.nominal_cols + self.numerical_cols

        self.ytrain = self.train_df[self.target].reset_index(drop=True)
        self.ytest = self.test_df[self.target].reset_index(drop=True)
        self.Xtrain = self.train_df[self.feature_cols].copy()
        self.Xtest = self.test_df[self.feature_cols].copy()

        self.num_qubits = len(self.feature_cols)
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.circuit = qml.QNode(self.qelm_circuit, self.dev)

        # Reservoir hyperparameters
        self.circuit_layers = 1     # 1 Trotter step: more layers scramble the state (thermalize) and erase input info
        self.J = -1                 # ZZ coupling
        self.B_z = 1.5                # longitudinal field
        self.B_x = 0.7                # transverse field (non-commuting term). Negligible at 1 layer/dt=0.1; keep for the physics story
        self.dt = 0.1               # Trotter step size

        self.nominal_maps = {}
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))

    # Encoding
    def fit_nominal_encoders(self):
        df = self.train_df
        for col in self.nominal_cols:
            order = df.groupby(col)[self.target].mean().sort_values().index
            self.nominal_maps[col] = {cat: i for i, cat in enumerate(order)}

    def apply_nominal_encoders(self, X):
        X = X.copy()
        for col in self.nominal_cols:
            mapping = self.nominal_maps[col]
            default_rank = (len(mapping) - 1) / 2.0          # unseen categories -> middle rank
            X[col] = X[col].map(mapping).fillna(default_rank).astype(float)
        return X

    def data_preprocessing(self):
        self.fit_nominal_encoders()
        self.Xtrain = self.apply_nominal_encoders(self.Xtrain)
        self.Xtest = self.apply_nominal_encoders(self.Xtest)

        self.Xtrain = self.scaler.fit_transform(self.Xtrain)
        self.Xtest = self.scaler.transform(self.Xtest)

    def angle_encoding(self, x_vector):
        qml.AngleEmbedding(features=x_vector, wires=range(self.num_qubits), rotation='Y')

    # Reservoir: Trotterized transverse-field Ising model
    def ising_reservoir(self):
        for _ in range(self.circuit_layers):
            # ZZ coupling
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-2 * self.J * self.dt, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            # longitudinal field B_z * Z
            for i in range(self.num_qubits):
                qml.RZ(-2 * self.B_z * self.dt, wires=i)
            # transverse field B_x * X  (non-commuting)
            if self.B_x != 0:
                for i in range(self.num_qubits):
                    qml.RX(-2 * self.B_x * self.dt, wires=i)

    def qelm_circuit(self, x_vector):
        self.angle_encoding(x_vector)
        self.ising_reservoir()

        observables = [qml.PauliZ(i) for i in range(self.num_qubits)]
        for i in range(self.num_qubits - 1):
            for j in range(i + 1, self.num_qubits):
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
        return [qml.expval(obs) for obs in observables]

    def quantum_transform(self, X):
        return np.array([self.circuit(row) for row in X])

    def _feature_names(self):
        single = [f"z_{i}" for i in range(self.num_qubits)]
        pairs = [f"zz_{i}_{j}" for i in range(self.num_qubits - 1)
                 for j in range(i + 1, self.num_qubits)]
        return single + pairs

    def run(self):
        train_path = OUTPUT_DIRECTORY / "loan_data_qelm_ising_train.csv"
        test_path = OUTPUT_DIRECTORY / "loan_data_qelm_ising_test.csv"

        if train_path.exists() and test_path.exists():
            print(f"QELM output already exists, skipping quantum simulation: {train_path.name}, {test_path.name}")
            return pd.read_csv(train_path), pd.read_csv(test_path)

        self.data_preprocessing()
        Xtrain_quantum = self.quantum_transform(self.Xtrain)
        Xtest_quantum = self.quantum_transform(self.Xtest)

        cols = self._feature_names()
        train_out = pd.DataFrame(Xtrain_quantum, columns=cols)
        train_out[self.target] = self.ytrain.values
        test_out = pd.DataFrame(Xtest_quantum, columns=cols)
        test_out[self.target] = self.ytest.values

        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)

        print(f"qubits={self.num_qubits}  train_features={Xtrain_quantum.shape}  test_features={Xtest_quantum.shape}")

        return train_out, test_out


if __name__ == "__main__":
    train_out, test_out = QELMIsing().run()
