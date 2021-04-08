package agent;

import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDList;

/**
 * Allows instantiation of a HexGame player.
 */
public class HexGamePlayer implements RlAgent {
	@Override
	public NDList chooseAction(RlEnv rlEnv, boolean b) {
		return null;
	}

	@Override
	public void trainBatch(RlEnv.Step[] steps) {

	}
}
