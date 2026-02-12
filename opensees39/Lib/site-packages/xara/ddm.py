class Domain:
    def __init__(self, partitions):
        from multiprocessing import Process, Queue

        self.num_parts = len(partitions)
        self.workers = []
        self.sub_messages = []
        self.sub_responses = []
        self.interface_guess = {}  # {(node, dof): value} for Dirichlet

        for part in partitions:
            message = Queue()
            response = Queue()

            # Start a new child process which will construct a model.
            p = Process(target=self._subdomain_worker, args=(part, message, response))
            p.start()

            self.workers.append(p)
            self.sub_messages.append(message)
            self.sub_responses.append(response)

    def _subdomain_worker(self, partition, messageueue, responseueue):
        sd = Subdomain(partition)
        while True:
            msg = messageueue.get()
            if msg == "TERMINATE":
                break

            elif msg["cmd"] == "solve":
                data = msg["data"]
                result = sd.solve_local(data)
                responseueue.put(result)

            elif msg["cmd"] == "schur":
                S, g = sd.get_schur_data()
                responseueue.put((S, g))

    def step(self):
        """
        Perform a single iteration:
        - Send current interface Dirichlet guess to all subdomains
        - Collect reactions
        - Return aggregate response (could be used to update guesses)
        """
        interface_data = {
            "dirichlet": self.interface_guess.copy(),
            "neumann": {}  # Could also support this later
        }

        # Send solve command with interface data to each subdomain
        for message in self.sub_messages:
            message.put({"cmd": "solve", "data": interface_data})

        # Collect responses
        all_reactions = []
        for response in self.sub_responses:
            result = response.get()
            reactions = result.get("reactions", {})
            all_reactions.append(reactions)

        # Optional: aggregate or store reactions for update
        return all_reactions

    def shutdown(self):
        for q in self.sub_messages:
            q.put("TERMINATE")
        for p in self.workers:
            p.join()

    def schur_update(self):
        S_total = None
        g_total = None

        # Request Schur data from all subdomains
        for message in self.sub_messages:
            message.put({"cmd": "schur"})

        for response in self.sub_responses:
            S_i, g_i = response.get()
            S_total = S_i if S_total is None else S_total + S_i
            g_total = g_i if g_total is None else g_total + g_i

        # Solve interface system
        u_gamma = np.linalg.solve(S_total, g_total)

        # Update interface guess
        # Map back into {(node,dof): value}
        self.interface_guess = self.unpack_interface_vector(u_gamma)




class Subdomain:
    def __init__(self, partition):
        import xara
#       self.model = xara.Model()
        import openseespy.opensees as ops

        model = ops
        self.partition = partition
        partition.populate(model)

        dofs = partition.get_dof_partition()
        self.interior_dofs  = dofs["interior"]
        self.interface_dofs = dofs["interface"]

    def solve_local(self, interface_data: dict):
        self.apply_interface_conditions(interface_data)
        self.model.analyze(1)
        return self.extract_interface_response()

    def apply_interface_conditions(self, interface_data: dict):
        dirichlet = interface_data.get("dirichlet", {})
        neumann = interface_data.get("neumann", {})

        for (node, dof), value in dirichlet.items():
            self.model.remove('sp', node, dof)  # Ensure clean slate
            self.model.fix(node, dof, value)

        for (node, dof), value in neumann.items():
            self.model.load(node, *self._vector_for_dof(dof, value))

    def extract_interface_response(self) -> dict:
        """Return reaction forces on interface DOFs."""
        responses = {"reactions": {}}
        for node in self.interface_dofs:
            for dof in self._active_dof_dirs(node):
                key = (node, dof)
                val = self.model.nodeReaction(node, dof)
                responses["reactions"][key] = val
        return responses

    def _vector_for_dof(self, dof: int, val: float):
        """Return a nodal load vector with `val` in the `dof`th position."""
        # Assuming 3 DOFs per node (can adapt)
        v = [0.0, 0.0, 0.0]
        v[dof-1] = val
        return v

    def _active_dof_dirs(self, node: int):
        # TODO: Replace with real DOF info from model or partition
        return [1, 2, 3]  # Assuming 3 translational DOFs

    def get_schur_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (S_i, g_i) for the interface DOFs.
        Assumes local tangent has been assembled.
        """
        # TODO(cmp) This should be done directly in OpenSees
        K = self.model.getTangent()
        R = self.model.getResidual()

        # Partition DOFs
        I = self.interior_dofs
        Γ = self.interface_dofs

        # Extract subblocks of K (using DOF maps)
        K_II = K[np.ix_(I, I)]
        K_IΓ = K[np.ix_(I, Γ)]
        K_ΓI = K_IΓ.T
        K_ΓΓ = K[np.ix_(Γ, Γ)]

        # Compute local Schur complement
        S = K_ΓΓ - K_ΓI @ np.linalg.solve(K_II, K_IΓ)

        # Compute local RHS
        f_I = R[I]
        f_Γ = R[Γ]
        g = f_Γ - K_ΓI @ np.linalg.solve(K_II, f_I)

        return S, g



