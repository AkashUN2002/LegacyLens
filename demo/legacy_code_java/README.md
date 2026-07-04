# Multi-Modal Evidence Review Pipeline (Java)

Java port of the Python LangGraph pipeline. Same function and variable names, same 14-column
output, same node/graph structure. Uses LangChain4j + Amazon Bedrock, Jackson, Commons CSV.

## Module map (Python -> Java)

| Python | Java |
|---|---|
| `state.py` | `src/main/java/com/claimreview/State.java` |
| `utility.py` | `src/main/java/com/claimreview/Utility.java` |
| `langgraph_workflow.py` | `src/main/java/com/claimreview/LangGraphWorkflow.java` + `StateGraph.java` + `StructuredLlm.java` |
| `main.py` | `src/main/java/com/claimreview/Main.java` |
| `evaluation/main.py` | `src/main/java/com/claimreview/evaluation/Main.java` |
| `prompts.ini` | `src/main/resources/prompts.ini` |
| `requirements.txt` | `pom.xml` |

## Build

```bash
cd java
mvn -q clean package
```

## Configure

Create `.env` in `java/` (or set env vars): `ANTHROPIC_MODEL`, `VISION_MODEL`,
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION=us-east-1`.

## Run

```bash
mvn exec:java                                                  # claims.csv -> output.csv
mvn exec:java -Dexec.args="../dataset/claims.csv out.csv"      # explicit paths
```

## Evaluate

```bash
mvn exec:java -Dexec.mainClass=com.claimreview.evaluation.Main
```
