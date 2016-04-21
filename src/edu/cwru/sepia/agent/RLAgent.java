package edu.cwru.sepia.agent;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

public class RLAgent extends Agent {

	/**
	 * Set in the constructor. Defines how many learning episodes your agent
	 * should run for. When starting an episode. If the count is greater than
	 * this value print a message and call sys.exit(0)
	 */
	public final int numEpisodes;

	/**
	 * List of your footmen and your enemies footmen
	 */
	private List<Integer> myFootmen;
	private List<Integer> enemyFootmen;
	private HashMap<Integer, Integer> footmenHealth;
	private HashMap<Integer, Position> footmenLocation;
	private boolean isExploitating;
	private HashMap<Integer, Double> staleQValue;
	private int episodeCount = 0;
	private boolean eventOccured = true;
	private double curReward = 0;
	private int curEpisode = 0;
	private List<Double> meanR;
	private int gamesWon = 0;
	private int gamesLost = 0;

	/**
	 * Convenience variable specifying enemy agent number. Use this whenever
	 * referring to the enemy agent. We will make sure it is set to the proper
	 * number when testing your code.
	 */
	public static final int ENEMY_PLAYERNUM = 1;

	/**
	 * Set this to whatever size your feature vector is.
	 */
	public static final int NUM_FEATURES = 6;

	/**
	 * Use this random number generator for your epsilon exploration. When you
	 * submit we will change this seed so make sure that your agent works for
	 * more than the default seed.
	 */
	public final Random random = new Random(12345);

	/**
	 * Your Q-function weights.
	 */
	public Double[] weights;

	/**
	 * These variables are set for you according to the assignment definition.
	 * You can change them, but it is not recommended. If you do change them
	 * please let us know and explain your reasoning for changing them.
	 */
	public final double gamma = 0.9;
	public final double learningRate = .0001;
	public final double epsilon = .02;

	public RLAgent(int playernum, String[] args) {
		super(playernum);

		if (args.length >= 1) {
			// numEpisodes = Integer.parseInt(args[0]);
			numEpisodes = 2001;
			System.out.println("Running " + numEpisodes + " episodes.");
		} else {
			numEpisodes = 10;
			System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
		}

		boolean loadWeights = false;
		if (args.length >= 2) {
			loadWeights = Boolean.parseBoolean(args[1]);
		} else {
			System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
		}

		if (loadWeights) {
			weights = loadWeights();
		} else {
			// initialize weights to random values between -1 and 1
			weights = new Double[NUM_FEATURES];
			for (int i = 0; i < weights.length; i++) {
				weights[i] = random.nextDouble() * 2 - 1;
			}
		}

		this.footmenHealth = new HashMap<Integer, Integer>();
		this.footmenLocation = new HashMap<Integer, Position>();
		this.staleQValue = new HashMap<Integer, Double>();
		this.meanR = new ArrayList<Double>();
	}

	/**
	 * We've implemented some setup code for your convenience. Change what you
	 * need to.
	 */
	@Override
	public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

		// You will need to add code to check if you are in a testing or
		// learning episode
		this.isExploitating = (this.episodeCount % 10) > 2;

		if (!this.isExploitating) {
			// if (this.average)
		}

		// Find all of your units
		myFootmen = new LinkedList<>();
		for (Integer unitId : stateView.getUnitIds(playernum)) {
			Unit.UnitView unit = stateView.getUnit(unitId);

			String unitName = unit.getTemplateView().getName().toLowerCase();
			if (unitName.equals("footman")) {
				myFootmen.add(unitId);
				this.footmenHealth.put(unitId, stateView.getUnit(unitId).getHP());
				this.footmenLocation.put(unitId, new Position(stateView.getUnit(unitId).getXPosition(),
						stateView.getUnit(unitId).getYPosition()));
				this.staleQValue.put(unitId, random.nextDouble());
			} else {
				System.err.println("Unknown unit type: " + unitName);
			}
		}

		// Find all of the enemy units
		enemyFootmen = new LinkedList<>();
		for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
			Unit.UnitView unit = stateView.getUnit(unitId);

			String unitName = unit.getTemplateView().getName().toLowerCase();
			if (unitName.equals("footman")) {
				enemyFootmen.add(unitId);
				this.footmenHealth.put(unitId, stateView.getUnit(unitId).getHP());
				this.footmenLocation.put(unitId, new Position(stateView.getUnit(unitId).getXPosition(),
						stateView.getUnit(unitId).getYPosition()));
			} else {
				System.err.println("Unknown unit type: " + unitName);
			}
		}

		return middleStep(stateView, historyView);
	}

	/**
	 * You will need to calculate the reward at each step and update your
	 * totals. You will also need to check if an event has occurred. If it has
	 * then you will need to update your weights and select a new action.
	 *
	 * If you are using the footmen vectors you will also need to remove killed
	 * units. To do so use the historyView to get a DeathLog. Each DeathLog
	 * tells you which player's unit died and the unit ID of the dead unit. To
	 * get the deaths from the last turn do something similar to the following
	 * snippet. Please be aware that on the first turn you should not call this
	 * as you will get nothing back.
	 *
	 * for(DeathLog deathLog :
	 * historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
	 * System.out.println("Player: " + deathLog.getController() + " unit: " +
	 * deathLog.getDeadUnitID()); }
	 *
	 * You should also check for completed actions using the history view.
	 * Obviously you never want a footman just sitting around doing nothing (the
	 * enemy certainly isn't going to stop attacking). So at the minimum you
	 * will have an even whenever one your footmen's targets is killed or an
	 * action fails. Actions may fail if the target is surrounded or the unit
	 * cannot find a path to the unit. To get the action results from the
	 * previous turn you can do something similar to the following. Please be
	 * aware that on the first turn you should not call this
	 *
	 * Map<Integer, ActionResult> actionResults =
	 * historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
	 * for(ActionResult result : actionResults.values()) {
	 * System.out.println(result.toString()); }
	 *
	 * @return New actions to execute or nothing if an event has not occurred.
	 */
	@Override
	public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {

		updateBasedOnEvent(stateView, historyView);
		Map<Integer, Action> returnActions = new HashMap<Integer, Action>();

		if (!this.eventOccured) {
			returnActions = assignAction(stateView, historyView, getAvailableFootmen(stateView, historyView));
		}
		if (this.eventOccured) {
			returnActions = assignAction(stateView, historyView, this.myFootmen);
			for (Integer footman : returnActions.keySet()) {
				this.curReward += this.calculateReward(stateView, historyView, footman);

				if (!this.isExploitating) {
					double[] f = this.calculateFeatureVector(stateView, historyView, footman,
							returnActions.get(footman).getUnitId());
					double staleQValue = this.staleQValue.get(footman);
					double[] fVal = this.calculateFeatureVector(stateView, historyView, footman,
							returnActions.get(footman).getUnitId());
					double updatedQValue = this.calcQValue(fVal);
					this.staleQValue.put(footman, updatedQValue);
					updateWeights(staleQValue, updatedQValue, f);
				}
			}
		}

		return returnActions;
	}

	private double[] updateWeights(double staleQVal, double updatedQVal, double[] features) {

		double[] doubleVals = new double[this.weights.length];

		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] = this.weights[i]
					+ this.learningRate * (this.curReward + this.gamma * updatedQVal - staleQVal) * features[i];
			doubleVals[i] = this.weights[i];
		}

		return doubleVals;
	}

	/**
	 * Here you will calculate the cumulative average rewards for your testing
	 * episodes. If you have just finished a set of test episodes you will call
	 * out testEpisode.
	 *
	 * It is also a good idea to save your weights with the saveWeights
	 * function.
	 */
	@Override
	public void terminalStep(State.StateView stateView, History.HistoryView historyView) {

		// MAKE SURE YOU CALL printTestData after you finish a test episode.
		updateBasedOnEvent(stateView, historyView);
		if (this.myFootmen.size() == 0) {
			System.out.println(this.curEpisode + " I LOST");
			this.gamesLost++;
		} else {
			System.out.println(this.curEpisode + " I WON");
			this.gamesWon++;
		}

		this.curEpisode++;
		if (this.curEpisode == this.numEpisodes) {
			printTestData(this.meanR);
			this.saveWeights(weights);
			System.out.println("games won: " + this.gamesWon);
			System.exit(0);
		}
		// Save your weights
		saveWeights(weights);
	}

	/**
	 * Given a footman and the current state and history of the game select the
	 * enemy that this unit should attack. This is where you would do the
	 * epsilon-greedy action selection.
	 *
	 * @param stateView
	 *            Current state of the game
	 * @param historyView
	 *            The entire history of this episode
	 * @param attackerId
	 *            The footman that will be attacking
	 * @return The enemy footman ID this unit should attack
	 */
	public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {

		boolean getRandom = (random.nextDouble() >= (1 - this.epsilon));

		if (getRandom) {
			return this.enemyFootmen.get(random.nextInt((enemyFootmen.size())));
		} else {
			return this.optimalEnemyToAttack(stateView, historyView, attackerId);
		}
	}

	/**
	 * Given the current state and the footman in question calculate the reward
	 * received on the last turn. This is where you will check for things like
	 * Did this footman take or give damage? Did this footman die or kill its
	 * enemy. Did this footman start an action on the last turn? See the
	 * assignment description for the full list of rewards.
	 *
	 * Remember that you will need to discount this reward based on the timestep
	 * it is received on. See the assignment description for more details.
	 *
	 * As part of the reward you will need to calculate if any of the units have
	 * taken damage. You can use the history view to get a list of damages dealt
	 * in the previous turn. Use something like the following.
	 *
	 * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
	 * System.out.println("Defending player: " +
	 * damageLog.getDefenderController() + " defending unit: " + \
	 * damageLog.getDefenderID() + " attacking player: " +
	 * damageLog.getAttackerController() + \ "attacking unit: " +
	 * damageLog.getAttackerID()); }
	 *
	 * You will do something similar for the deaths. See the middle step
	 * documentation for a snippet showing how to use the deathLogs.
	 *
	 * To see if a command was issued you can check the commands issued log.
	 *
	 * Map<Integer, Action> commandsIssued =
	 * historyView.getCommandsIssued(playernum, lastTurnNumber); for
	 * (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
	 * System.out.println("Unit " + commandEntry.getKey() + " was command to " +
	 * commandEntry.getValue().toString); }
	 *
	 * @param stateView
	 *            The current state of the game.
	 * @param historyView
	 *            History of the episode up until this turn.
	 * @param footmanId
	 *            The footman ID you are looking for the reward from.
	 * @return The current reward
	 */
	public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
		double reward = -1;
		int turn = stateView.getTurnNumber();

		if (turn > 0) {
			for (DamageLog damage : historyView.getDamageLogs(turn - 1)) {
				if (damage.getDefenderController() == this.playernum) {
					reward -= damage.getDamage();
				}
				if (damage.getDefenderController() == ENEMY_PLAYERNUM) {
					reward += damage.getDamage();
				}
			}
			for (DeathLog death : historyView.getDeathLogs(turn - 1)) {
				if (death.getController() == this.playernum) {
					for (Integer footmenID : this.myFootmen) {
						if (death.getDeadUnitID() == footmenID) {
							reward -= 100;
						}
					}
				}
				if (death.getController() == ENEMY_PLAYERNUM) {
					for (Integer enemyID : this.enemyFootmen) {
						if (death.getDeadUnitID() == enemyID) {
							reward += 100;
						}
					}
				}
			}
		}
		return reward;
	}

	/**
	 * Calculate the Q-Value for a given state action pair. The state in this
	 * scenario is the current state view and the history of this episode. The
	 * action is the attacker and the enemy pair for the SEPIA attack action.
	 *
	 * This returns the Q-value according to your feature approximation. This is
	 * where you will calculate your features and multiply them by your current
	 * weights to get the approximate Q-value.
	 *
	 * @param stateView
	 *            Current SEPIA state
	 * @param historyView
	 *            Episode history up to this point in the game
	 * @param attackerId
	 *            Your footman. The one doing the attacking.
	 * @param defenderId
	 *            An enemy footman that your footman would be attacking
	 * @return The approximate Q-value
	 */
	public double calcQValue(State.StateView stateView, History.HistoryView historyView, int attackerId,
			int defenderId) {
		double[] f = this.calculateFeatureVector(stateView, historyView, attackerId, defenderId);
		double sum = 0;
		for (int i = 0; i < f.length; i++) {
			sum += f[i] * this.weights[i];
		}
		return sum;
	}

	private double calcQValue(double[] features) {
		double sum = 0;
		for (int i = 0; i < features.length; i++) {
			sum += features[i] * this.weights[i];
		}
		return sum;
	}

	/**
	 * Given a state and action calculate your features here. Please include a
	 * comment explaining what features you chose and why you chose them.
	 *
	 * All of your feature functions should evaluate to a double. Collect all of
	 * these into an array. You will take a dot product of this array with the
	 * weights array to get a Q-value for a given state action.
	 *
	 * It is a good idea to make the first value in your array a constant. This
	 * just helps remove any offset from 0 in the Q-function. The other features
	 * are up to you. Many are suggested in the assignment description.
	 *
	 * @param stateView
	 *            Current state of the SEPIA game
	 * @param historyView
	 *            History of the game up until this turn
	 * @param attackerId
	 *            Your footman. The one doing the attacking.
	 * @param defenderId
	 *            An enemy footman. The one you are considering attacking.
	 * @return The array of feature function outputs.
	 */
	public double[] calculateFeatureVector(State.StateView stateView, History.HistoryView historyView, int attackerId,
			int defenderId) {

		double[] featureVector = new double[NUM_FEATURES];

		featureVector[0] = 1;
		featureVector[1] = this.footmenHealth.get(attackerId);
		featureVector[2] = this.footmenHealth.get(defenderId);
		if (this.footmenHealth.get(attackerId) == 0) {
			featureVector[3] = this.footmenHealth.get(defenderId) / .5d;
		} else {
			featureVector[3] = this.footmenHealth.get(defenderId) / this.footmenHealth.get(attackerId);
		}
		featureVector[4] = isEnemyNextToMe(defenderId, attackerId) ? 0.5 : 0;
		featureVector[5] = surroundingEnemies(defenderId);
		return featureVector;
	}

	private Integer optimalEnemyToAttack(State.StateView state, History.HistoryView history, int attackerId) {
		int optimalEnemyID = this.enemyFootmen.get(0);
		double optimalQVal = Double.NEGATIVE_INFINITY;

		for (Integer enemy : this.enemyFootmen) {
			double[] f = this.calculateFeatureVector(state, history, attackerId, enemy);
			double qVal = this.calcQValue(f);
			if (qVal > optimalQVal) {
				optimalEnemyID = enemy;
				optimalQVal = qVal;
			}
		}
		return optimalEnemyID;
	}

	private boolean isEnemyNextToMe(int myFootmenID, int enemyID) {
		return this.footmenLocation.get(myFootmenID).isAdjacent(this.footmenLocation.get(enemyID));
	}

	private int surroundingEnemies(int myFootmenID) {
		int count = 0;
		for (Integer enemyID : this.enemyFootmen) {
			Position p1 = this.footmenLocation.get(myFootmenID);
			Position p2 = this.footmenLocation.get(enemyID);
			if (this.footmenLocation.get(myFootmenID).isAdjacent(this.footmenLocation.get(enemyID))) {
				count++;
			}
		}
		return count;
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * Prints the learning rate data described in the assignment. Do not modify
	 * this method.
	 *
	 * @param averageRewards
	 *            List of cumulative average rewards from test episodes.
	 */
	public void printTestData(List<Double> averageRewards) {
		System.out.println("");
		System.out.println("Games Played      Average Cumulative Reward");
		System.out.println("-------------     -------------------------");
		for (int i = 0; i < averageRewards.size(); i++) {
			String gamesPlayed = Integer.toString(10 * i);
			String averageReward = String.format("%.2f", averageRewards.get(i));

			int numSpaces = "-------------     ".length() - gamesPlayed.length();
			StringBuffer spaceBuffer = new StringBuffer(numSpaces);
			for (int j = 0; j < numSpaces; j++) {
				spaceBuffer.append(" ");
			}
			System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
		}
		System.out.println("");
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * This function will take your set of weights and save them to a file.
	 * Overwriting whatever file is currently there. You will use this when
	 * training your agents. You will include th output of this function from
	 * your trained agent with your submission.
	 *
	 * Look in the agent_weights folder for the output.
	 *
	 * @param weights
	 *            Array of weights
	 */
	public void saveWeights(Double[] weights) {
		File path = new File("agent_weights/weights.txt");
		// create the directories if they do not already exist
		path.getAbsoluteFile().getParentFile().mkdirs();

		try {
			// open a new file writer. Set append to false
			BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

			for (double weight : weights) {
				writer.write(String.format("%f\n", weight));
			}
			writer.flush();
			writer.close();
		} catch (IOException ex) {
			System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
		}
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * This function will load the weights stored at agent_weights/weights.txt.
	 * The contents of this file can be created using the saveWeights function.
	 * You will use this function if the load weights argument of the agent is
	 * set to 1.
	 *
	 * @return The array of weights
	 */
	public Double[] loadWeights() {
		File path = new File("agent_weights/weights.txt");
		if (!path.exists()) {
			System.err.println("Failed to load weights. File does not exist");
			return null;
		}

		try {
			BufferedReader reader = new BufferedReader(new FileReader(path));
			String line;
			List<Double> weights = new LinkedList<>();
			while ((line = reader.readLine()) != null) {
				weights.add(Double.parseDouble(line));
			}
			reader.close();

			return weights.toArray(new Double[weights.size()]);
		} catch (IOException ex) {
			System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
		}
		return null;
	}

	@Override
	public void savePlayerData(OutputStream outputStream) {

	}

	@Override
	public void loadPlayerData(InputStream inputStream) {

	}

	private void updateBasedOnEvent(State.StateView state, History.HistoryView history) {

		for (Unit.UnitView unit : state.getAllUnits()) {
			this.footmenHealth.put(unit.getID(), unit.getHP());
			this.footmenLocation.put(unit.getID(), new Position(unit.getXPosition(), unit.getYPosition()));
		}

		if (state.getTurnNumber() <= 0) {
			eventOccured = true;
			return;
		} else if (history.getDeathLogs(state.getTurnNumber() - 1).size() > 0) {
			eventOccured = true;
		}
		for (DeathLog death : history.getDeathLogs(state.getTurnNumber() - 1)) {
			if (death.getController() == ENEMY_PLAYERNUM) {
				this.enemyFootmen.remove(enemyFootmen.indexOf(death.getDeadUnitID()));
			} else if (death.getController() == this.playernum) {
				this.myFootmen.remove(myFootmen.indexOf(death.getDeadUnitID()));
			}
		}
		for (DamageLog damage : history.getDamageLogs(state.getTurnNumber() - 1)) {
			if (damage.getDefenderController() == playernum) {
				eventOccured = true;
				return;
			}
		}
		eventOccured = false;
	}

	private List<Integer> getAvailableFootmen(State.StateView state, History.HistoryView history) {
		ArrayList<Integer> idleFootmenID = new ArrayList<Integer>();
		Map<Integer, ActionResult> results = history.getCommandFeedback(playernum, state.getTurnNumber() - 1);
		for (ActionResult action : results.values()) {
			if (action.getFeedback() == ActionFeedback.COMPLETED) {
				idleFootmenID.add(action.getAction().getUnitId());
			}
		}
		return idleFootmenID;
	}

	private Map<Integer, Action> assignAction(State.StateView state, History.HistoryView history,
			List<Integer> myFootmen) {
		HashMap<Integer, Action> actionPairs = new HashMap<Integer, Action>();

		for (Integer footman : myFootmen) {
			actionPairs.put(footman, Action.createCompoundAttack(footman, this.selectAction(state, history, footman)));
		}

		return actionPairs;
	}
}
