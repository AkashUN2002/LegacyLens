package com.claimreview;

import com.claimreview.State.ClaimState;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Minimal LangGraph-style state machine: nodes return partial state updates that are
 * merged into a single flat ClaimState. Supports plain edges and one conditional edge.
 */
public class StateGraph {

    public static final String START = "__start__";
    public static final String END = "__end__";

    private final Map<String, Function<ClaimState, Map<String, Object>>> nodes = new LinkedHashMap<>();
    private final Map<String, String> edges = new LinkedHashMap<>();
    private String condFrom;
    private Function<ClaimState, String> condFn;
    private Map<String, String> condMap;

    public void add_node(String node_name, Function<ClaimState, Map<String, Object>> node_function) {
        nodes.put(node_name, node_function);
    }

    public void add_edge(String from, String to) {
        edges.put(from, to);
    }

    public void add_conditional_edges(String from, Function<ClaimState, String> router, Map<String, String> mapping) {
        this.condFrom = from;
        this.condFn = router;
        this.condMap = mapping;
    }

    public StateGraph compile() {
        return this;
    }

    public ClaimState invoke(Map<String, Object> payload) {
        ClaimState state = new ClaimState();
        state.putAll(payload);
        String current = edges.get(START);
        while (current != null && !current.equals(END)) {
            Map<String, Object> update = nodes.get(current).apply(state);
            if (update != null) state.putAll(update);
            if (current.equals(condFrom)) {
                current = condMap.get(condFn.apply(state));
            } else {
                current = edges.get(current);
            }
        }
        return state;
    }

    public static List<Object> asList(Object o) {
        return o == null ? new ArrayList<>() : (List<Object>) o;
    }
}
