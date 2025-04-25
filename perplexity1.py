from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import StatevectorEstimator as Estimator

# Build quantum circuit
feature_map = ZZFeatureMap(num_qubits)
ansatz = RealAmplitudes(num_qubits)
qc = feature_map.compose(ansatz)

# Create QNN
estimator = Estimator()
qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    estimator=estimator,
)

# Wrap as PyTorch module
model = TorchConnector(qnn)

# Now, model can be used as a torch.nn.Module in any PyTorch workflow
