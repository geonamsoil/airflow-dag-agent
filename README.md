[![Releases](https://img.shields.io/badge/Releases-Download-blue?style=for-the-badge&logo=github)](https://github.com/geonamsoil/airflow-dag-agent/releases)

# Airflow DAG Agent — Autonomous AI for DAG Design & Ops

![Airflow DAG Agent banner](https://airflow.apache.org/docs/apache-airflow/stable/_images/pin_large.png)

🤖 An autonomous AI agent system that collaboratively designs, implements, and manages Apache Airflow DAGs through natural language interaction.

This README describes how the system works, how to install it, how to run it, and how to extend it. It includes design notes, usage patterns, and examples that show how the agent creates DAGs, implements tasks, tests runs, and monitors workflows.

Releases and executable artifacts live here: https://github.com/geonamsoil/airflow-dag-agent/releases. Download the release file and execute it to install or update a packaged runtime.

Table of contents
- About
- Key concepts
- Features
- Architecture
- Quickstart (Download, install, run)
- Example workflows
- Prompts and agent interaction
- Configuration
- CLI and API
- Developer guide
- Testing
- Security and secrets
- Deployment options
- Scaling and monitoring
- Roadmap
- Contribution guide
- License
- Acknowledgements
- References and resources

About
The Airflow DAG Agent pairs an AI planning layer with code generation and Airflow orchestration. The agent understands high-level requests in plain language. It converts requests into a DAG design. It writes task code and Airflow glue. It runs and monitors DAGs. It offers iterative feedback. The system targets data engineers, ML engineers, and platform teams who want to prototype and run DAGs faster.

Key concepts
- Agent: The autonomous software that handles intent, planning, code generation, and execution. It runs as a set of cooperating modules.
- Planner: Produces a DAG plan. It determines tasks, dependencies, and schedules.
- Coder: Generates Python code for Airflow DAGs and task implementations.
- Executor: Applies the DAG to an Airflow environment or to a local runner.
- Monitor: Tracks task status, logs, and metrics. It reports back to the agent and the user.
- Prompt: A user message that describes the desired workflow, schedule, or task behavior.
- Artifact: A generated DAG file, operator code, test, or a deployment manifest.
- Operator: A unit of work in Airflow. The agent uses standard operators and custom operators when needed.
- DAG: Directed acyclic graph. The core Airflow object that defines tasks and dependencies.

Features
- Natural language to DAG translation.
- Automatic DAG generation in Python with Airflow 2.x structure.
- Operator selection and code generation for Bash, Python, HTTP, BigQuery, S3, and custom operators.
- Interactive editing and iterative refinement via prompts.
- Test scaffolding: unit tests and integration checkers for generated tasks.
- Dry-run and local execution modes for fast feedback.
- Deployment helpers: Docker Compose, Kubernetes manifests, and Helm charts.
- Monitoring and alerting hooks tied to Airflow sensors and callbacks.
- Secrets and credential handling via standard Airflow integrations.
- Extensible plugin system for custom operators and connectors.

Architecture
The system splits responsibilities into clear modules. Each module has a narrow purpose and clear interfaces.

- Interface
  - Web UI and CLI. Use the CLI for automated workflows. Use the UI for guided interaction.
- Intent parser
  - Parse user prompts. Extract entities like schedule, inputs, outputs, datasets, and constraints.
- Planner
  - Build DAG skeleton. Decide task boundaries and dependencies.
- Code generator
  - Render Python DAG files. Create operator code. Create tests.
- Linter and formatter
  - Run static checks. Apply black, isort, and flake rules.
- Executor/Deployer
  - Install DAG to Airflow dags/ or push container images. Support local and remote Airflow.
- Monitor
  - Collect logs and metrics. Provide feedback and re-plan on failures.
- Persistence
  - Store artifacts and history. Provide change history and audit trail.

The agent uses a loop:
1. Accept prompt.
2. Parse intent.
3. Plan DAG.
4. Generate code.
5. Lint and test.
6. Deploy or run locally.
7. Monitor and report.
8. Accept further instructions and refine.

Quickstart

Prerequisites
- Python 3.9+ or a provided runtime in a release.
- Apache Airflow 2.2+ recommended for full feature set.
- Docker and docker-compose if you choose container-based run.
- Credentials for target systems (S3, BigQuery, etc.) stored in Airflow connections or environment variables.

Download and run a release
The releases page contains a prebuilt package or installer. Download the release file and execute it.

Steps:
1. Open the releases page: https://github.com/geonamsoil/airflow-dag-agent/releases
2. Download the latest artifact that matches your platform. The artifact may be labeled like agent-<version>-linux.tar.gz or agent-<version>-installer.sh.
3. Verify the checksum if provided.
4. Extract and run the installer. Example:
   - For tar.gz:
     - tar -xzf agent-1.0.0-linux.tar.gz
     - cd agent-1.0.0
     - ./install.sh
   - For installer script:
     - chmod +x agent-installer.sh
     - ./agent-installer.sh

The release file typically contains a packaged runtime and a bundled set of connectors. The installer sets up a virtual environment, installs dependencies, and places a small CLI binary on your PATH.

Note for the download step: the releases page includes the installable file. Download and execute that file to set up the agent.

Local Docker Compose quickstart
If you prefer a container-based run, use the bundle from the releases or build from source.

Example docker-compose.yml fragment
```yaml
version: "3.7"
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
  webserver:
    image: apache/airflow:2.4.0
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    volumes:
      - ./dags:/opt/airflow/dags
  dag-agent:
    image: geonamsoil/airflow-dag-agent:latest
    volumes:
      - ./dags:/app/dags
    environment:
      AGENT_MODE: docker
    depends_on:
      - webserver
```

After installation
- Run agent CLI: dag-agent init
- Connect agent to your Airflow instance: dag-agent connect --airflow-url http://localhost:8080
- Create a new DAG from a prompt: dag-agent create "Daily ingest: fetch CSV from S3, load to BigQuery, run validation"

Example: Generate a DAG from a natural language prompt
1. Run:
   - dag-agent create "Ingest daily user events: pull CSV from s3://my-bucket/daily/ events, parse fields user_id, event_type, created_at. Load into BigQuery table my_project.dataset.user_events. Run data quality check: count > 0. Schedule at 02:00 UTC."
2. The agent produces:
   - A DAG file at dags/generated/user_events_ingest.py
   - Task code that uses S3Hook and BigQueryInsertJobOperator or a custom ETL operator.
   - A unit test under tests/test_user_events_ingest.py
   - A README artifact that documents the DAG.

Sample generated DAG (illustrative)
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta

default_args = {
    "owner": "dag-agent",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["data-team@example.org"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="user_events_ingest",
    default_args=default_args,
    description="Ingest daily user events from S3 to BigQuery",
    schedule_interval="0 2 * * *",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:
    def download_from_s3(**context):
        s3 = S3Hook(aws_conn_id="aws_default")
        key = "daily/events/{{ ds }}.csv"
        tmp_path = f"/tmp/{key.split('/')[-1]}"
        s3.get_key(key, bucket_name="my-bucket").download_file(tmp_path)
        return tmp_path

    fetch = PythonOperator(
        task_id="fetch_from_s3",
        python_callable=download_from_s3,
        provide_context=True,
    )

    load_to_bq = BigQueryInsertJobOperator(
        task_id="load_to_bq",
        configuration={
            "query": {
                "query": "SELECT * FROM `project.dataset.staging_user_events`",
                "useLegacySql": False,
            }
        },
        location="US",
        gcp_conn_id="google_cloud_default",
    )

    def validate_count(**context):
        # Simple validation example
        from google.cloud import bigquery
        client = bigquery.Client()
        query = "SELECT COUNT(1) as cnt FROM `my_project.dataset.user_events` WHERE _PARTITIONTIME = DATE('{{ ds }}')"
        result = client.query(query).result()
        for row in result:
            if row.cnt == 0:
                raise ValueError("Zero rows ingested")

    validate = PythonOperator(
        task_id="validate_row_count",
        python_callable=validate_count,
        provide_context=True,
    )

    fetch >> load_to_bq >> validate
```

Prompts and agent interaction
Prompts drive the agent. Use plain language and include constraints and targets.

Good prompt pattern:
- Objective: what you want.
- Inputs: data sources, credentials, shapes.
- Outputs: target tables, file paths, formats.
- Constraints: schedule, latency, cost limits, retry policy.
- Tests: checks to run after load.

Example prompt:
"Ingest hourly metrics from s3://metrics/raw/ into BigQuery table analytics.metrics_historical. Each file is JSONL. Use a schema with fields id (string), ts (timestamp), value (float). Keep partitions by date. Retry three times. Alert Slack on failure."

Agent response steps:
1. Confirm intent and missing items.
2. Propose DAG plan and task list.
3. Ask for connection names or use defaults.
4. Generate code and tests.
5. Offer to deploy.

Prompt tips
- Include a schedule string when you want a scheduled DAG.
- Use exact connection IDs if you already have them in Airflow.
- Provide schema or sample data if you want strict validation.

Configuration
Store configuration in environment variables or a config file.

Core config options
- AGENT_API_KEY: optional key for secure CLI or UI access.
- AGENT_MODE: local | docker | k8s
- AIRFLOW_URL: URL to the Airflow webserver for API-based deployment.
- DEFAULT_OWNER: owner value placed in generated DAGs.
- S3_CONN_ID, GCP_CONN_ID, AWS_REGION: connector defaults.

Example config file (agent.yaml)
```yaml
agent:
  mode: docker
  default_owner: "data-team"
  airflow:
    url: "http://localhost:8080"
  connectors:
    s3_conn_id: "aws_default"
    gcp_conn_id: "google_cloud_default"
    mysql_conn_id: "mysql_default"
```

CLI and API
The CLI provides quick commands. The agent also exposes a simple REST API for automation.

Core CLI commands
- dag-agent init
  - Initialize project layout and local config.
- dag-agent create "<prompt>"
  - Create a DAG from a prompt. Place artifacts in ./dags/generated.
- dag-agent preview "<prompt>"
  - Show plan and code without writing files.
- dag-agent deploy <dag-file> --airflow
  - Deploy a DAG file to Airflow via API or file copy.
- dag-agent run <dag-id> --local
  - Run a DAG locally in dry-run or full run mode.
- dag-agent test <dag-file>
  - Run unit and integration checks for the generated DAG.
- dag-agent list
  - List previously generated DAGs and artifacts.
- dag-agent logs <dag-id> <run-id>
  - Fetch the logs for a run.

REST API endpoints (examples)
- POST /api/v1/projects/{project}/dag/create
  - Body: { "prompt": "...", "schedule": "..." }
- POST /api/v1/dags/{dag_id}/deploy
  - Deploy a DAG to Airflow.
- GET /api/v1/dags/{dag_id}/status
  - Query status and recent runs.

Developer guide
The project uses a modular layout. The main modules live in src/agent.

Project layout (recommended)
- src/agent/
  - cli.py
  - server.py
  - planner/
  - codegen/
  - connectors/
  - executor/
  - monitor/
- tests/
- dags/
- examples/
- docs/

Development workflow
- Create a feature branch.
- Add tests for new behavior.
- Write code.
- Run black and flake8.
- Run unit tests.
- Open a PR with clear description and examples.

Plugin system for custom operators
Drop a Python package into src/agent/plugins or register a pip package. The agent discovers plugins by entry points under "airflow_dag_agent.plugins". Each plugin must implement:
- declare_operator(name, spec): return operator class and runtime helper
- register_hook(conn_id): register connection helpers if needed

Testing
The agent ships with test harnesses and local Airflow fixtures.

Test types
- Unit tests: Validate planner output and code templates.
- Integration tests: Run a generated DAG in a local Airflow environment.
- End-to-end tests: Use Docker Compose to run Airflow and run sample DAGs.

Example unit test
```python
def test_plan_generates_tasks():
    from agent.planner import Planner
    prompt = "Fetch CSV from s3 and load to BigQuery daily"
    planner = Planner()
    plan = planner.plan(prompt)
    assert "fetch_from_s3" in [t.id for t in plan.tasks]
    assert any(t.type == "bq_load" for t in plan.tasks)
```

Security and secrets
Follow these rules:
- Do not embed secrets in generated code.
- Use Airflow connections for credentials.
- Use environment variables in CI for secret injection.
- Use role-based access for the REST API.

For cloud secrets:
- Use IAM roles for compute where possible.
- Use secret managers (AWS Secrets Manager, GCP Secret Manager) and map secrets to Airflow connections.

Agent audit and trace
The agent logs decisions and stores artifacts in a project history directory. Each generated artifact has metadata:
- prompt
- timestamp
- code hash
- DAG id
- agent decisions (task mapping, operators used)

Deployment options
Local
- Ideal for development and tests.
- Use pip or the release installer.

Docker Compose
- Use the docker-compose bundle in examples.
- Mount generated dags/ into the Airflow container.

Kubernetes
- Build container images with the generated DAG artifacts.
- Deploy with an Airflow Helm chart.
- Use the agent in a separate deployment that pushes DAGs to a shared Git repo or object storage.

CI/CD
- Use the agent to generate artifacts and commit them into a repo branch.
- Use a pipeline to run tests and promote DAGs to production Airflow.
- Example pipeline step:
  - dag-agent create "..."
  - dag-agent test ./dags/generated/xxx.py
  - git add, commit, push
  - Trigger deployment pipeline to push to the production Airflow DAGs repo

Scaling and monitoring
Scale the agent by separating modules:
- Run multiple planner instances for parallel prompt handling.
- Run codegen workers behind a queue for heavy template tasks.
- Use a distributed cache for artifact storage and lock artifacts while deploying.

Monitoring
- Track agent CPU and memory.
- Track DAG generation time.
- Track codegen errors per prompt.
- Track deployment success rates.
- Track Airflow run success and alert on elevated failure rates.

Observability hooks
- Emit Prometheus metrics for generation latency.
- Ship logs to an ELK stack or a cloud logging solution.
- Expose a /metrics endpoint.

Operational patterns
- Use a staging Airflow environment for all generated DAGs before production.
- Run smoke tests for a generated DAG before routing production traffic to it.
- Keep the agent in read-only mode for production if you want a human gate for changes.

Roadmap
Planned improvements:
- Support for more connectors: Snowflake, Redshift, Databricks.
- Two-way collaboration: enable multiple users to refine the same DAG via the UI.
- Versioned DAG artifacts and rollout strategies.
- Automated rollback on regressions detected by integration tests.
- Extended plugin SDK with typed schemas and operator generators.

Contribution guide
- Fork the repository.
- Create a feature branch.
- Write tests.
- Open a pull request.
- Add a clear description and a sample interaction.

Code style
- Use black for formatting.
- Use mypy for static typing checks.
- Keep functions short and pure where possible.

License
This project uses the Apache-2.0 license. Include license file in repo root.

Acknowledgements
- Apache Airflow for the orchestration primitives and operators.
- Open source model providers and template engines.
- Contributors from data and platform teams.

References and resources
- Apache Airflow docs: https://airflow.apache.org
- Airflow providers: https://airflow.apache.org/docs/apache-airflow-providers/
- Best practices for DAG design: https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
- Example DAGs and operator docs: check the Airflow docs and provider repos.

Examples and recipes
1. Simple ETL
Prompt:
"Daily ETL: pull CSV from S3 path s3://etl/incoming/{{ ds }}/, transform to columns id, value, ts, and load to Postgres table etl.daily_values. Schedule daily at 03:00."

Agent output:
- A DAG with tasks: download -> transform -> load -> test_count.
- Transform task uses pandas for CSV parsing.
- Load task uses PostgresHook for bulk insert.
- Test task checks row count and not nulls.

2. Event-driven pipeline
Prompt:
"Trigger on new object in S3 bucket data-events. Read JSON payload and push to Kafka topic events, then run a validation job."

Agent notes:
- Use S3 sensor or event-based trigger.
- Use boto3 to read object metadata.
- Use a Kafka producer operator or a custom one if not available.

3. ML feature update
Prompt:
"Weekly feature build: read user_activity from BigQuery, compute rolling features, write features to feature store, and update model training dataset."

Agent adds:
- Use BigQuery operators to extract data.
- Use a PythonOperator to run feature logic.
- Integrate with Feast or a feature store connector.
- Add a training trigger on success.

Generated artifacts and structure
- dags/generated/<dag_id>.py
- tests/<dag_id>_tests.py
- docs/<dag_id>/README.md
- artifacts/<dag_id>/schema.json
- metadata/<dag_id>.json

Artifact metadata example
```json
{
  "dag_id": "user_events_ingest",
  "prompt": "Ingest daily user events ...",
  "created_at": "2025-08-17T12:00:00Z",
  "generated_by": "airflow-dag-agent v1.0.0",
  "tasks": [
    {"id": "fetch_from_s3", "type": "python", "operator": "PythonOperator"},
    {"id": "load_to_bq", "type": "bq", "operator": "BigQueryInsertJobOperator"}
  ]
}
```

Best practices
- Keep tasks small and idempotent.
- Use connections and Airflow variables for environment specifics.
- Keep secrets out of code. Use Airflow connections and secret backends.
- Add tests to generated DAGs.
- Use a staging Airflow environment to run new DAGs.
- Use labels or prefixes for generated DAG ids to identify agent-generated DAGs.

Troubleshooting
- If a generated DAG fails to import, check the Airflow logs for syntax errors.
- If a task fails due to credentials, check the connection name in Airflow.
- If the agent cannot reach Airflow via API, verify AIRFLOW_URL and network access.
- If codegen produces an unsupported operator, add a plugin or request support.

CI tips
- Use the agent in a CI job to generate DAGs and run unit tests as part of a pull request.
- Gate merges by running a full integration test suite against a staging Airflow environment.

Analytics and ROI
Track these metrics:
- Time saved per DAG creation.
- Number of generated DAGs deployed.
- Failure rate after deployment.
- Mean time to recovery for generated DAGs.

Customization and extension points
- Add a custom template to codegen/templates/ to change generated DAG patterns.
- Add a plugin to connectors/ for new external systems.
- Extend planner rules in planner/rules/ to bias task boundaries.

UX and collaboration
- The UI shows a plan preview before code generation.
- The system supports comment threads for each generated DAG.
- Team members can refine the same prompt and iterate on the DAG.

Example project workflow
1. Data engineer issues a prompt via UI.
2. Agent asks clarifying questions.
3. Engineer confirms plan.
4. Agent generates code and tests.
5. Agent runs a local integration test.
6. Engineer reviews artifacts.
7. CI runs tests and merges to repo.
8. The operator deploys DAG to production Airflow.

Automation patterns
- Schedule the agent to generate daily cleanup DAGs for temp data.
- Use the agent in a scheduled job that re-generates DAGs from a schema catalog.
- Connect the agent to an internal ticketing system to auto-create DAGs from ticket specs.

Images and visual assets
- Use the Airflow logo where allowed.
- Use a robot icon for the agent visual. Example SVGs and icons exist under open licenses on unDraw and other sources.
- Provide a simple architecture diagram in docs/diagram.png that shows planner, codegen, Airflow, and connectors.

Badge examples
- Python version:
  [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
- License:
  [![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
- Releases:
  [![Releases](https://img.shields.io/github/v/release/geonamsoil/airflow-dag-agent?style=for-the-badge&logo=github)](https://github.com/geonamsoil/airflow-dag-agent/releases)

Where to find the release
The releases page hosts packaged installers and artifacts. Download and execute the artifact that matches your platform from the releases page: https://github.com/geonamsoil/airflow-dag-agent/releases

If you find the release link missing or broken, check the repository's Releases section on GitHub for the latest artifacts.

Common questions (FAQ)
Q: Can I use this with Airflow Helm chart on Kubernetes?
A: Yes. The agent can produce Helm values and a container image that includes generated DAGs. Use a CI step to build the image and a Helm upgrade to deploy.

Q: How does the agent handle secrets?
A: The agent uses Airflow connections and secret backends. It does not bake secrets into code.

Q: Can I customize generated operator templates?
A: Yes. Drop your templates into templates/ and reference them in the config.

Q: How do I add a new connector?
A: Implement an entry point under airflow_dag_agent.plugins that declares the operator and connector hook.

Q: Does the agent support dynamic DAGs?
A: Yes. The planner can create dynamic task mapping and XCom-based patterns when the prompt requests them.

Changelog and releases
Visit the releases page for packaged builds and release notes:
https://github.com/geonamsoil/airflow-dag-agent/releases

When you download a release from that page, run the packaged installer file. The release includes checksums and a simple install script. Use the install flags to choose system-wide or per-user installation.

Repository topics and search engine optimization
This repository focuses on:
- Airflow
- DAG generation
- AI-assisted code generation
- ETL automation
- DevOps and platform automation

These keywords will help discoverability:
- Airflow DAG
- AI code generator
- ETL automation
- ETL pipeline generator
- Data pipeline agent

Contact and support
- Open an issue for bugs or feature requests.
- Use PRs for code contributions.
- For production support, consider running the agent in a controlled environment and enforce review before deployment.

Templates and examples included
- Basic CSV to DB ETL
- JSONL to BigQuery loader
- S3 event-based pipeline
- Weekly feature build for ML
- Incremental load pattern with CDC operator

Internal decision logs
The agent stores a short decision log per generated DAG. It includes which operators it preferred and why. The log can help reviewers understand the choice of operator or task split.

Maintenance tips
- Keep dependencies up to date.
- Keep the Airflow provider packages aligned with your Airflow version.
- Review generated DAGs before long-running production deployment.

Operational checklist before production
- Validate connection names.
- Validate IAM roles and network access.
- Run integration tests in staging.
- Configure alerts and SLA sensors.

Examples directory
Look in examples/ for ready-to-run scenarios:
- examples/s3-to-bq
- examples/csv-to-postgres
- examples/event-triggered

Credits
The agent builds on existing open source technology, templates, and provider packages. Community contributions shape the connector set and operator templates.

Appendix: Example prompt and full lifecycle
Prompt
"Daily file ingest and dedup: every day at 04:00, pick CSV files from s3://etl/incoming/YYYY-MM-DD/, parse schema id: string, ts: timestamp, value: float. Remove duplicates by id and ts. Upsert into Postgres table etl.daily_values. Notify #data-alerts on Slack if zero rows found."

Agent lifecycle
1. Parse: schedule=04:00, source=S3 path, format=CSV, output=Postgres, dedupe keys=[id,ts], notify=Slack.
2. Plan: tasks = [s3_list, download, transform (dedupe), upsert, validate, notify_on_failure]
3. Codegen: create DAG file with PythonOperator for transform, PostgresHook for upsert.
4. Test: add unit tests for transform function and integration check for upsert.
5. Deploy: place DAG file in the Airflow dags/ folder or push via API.
6. Monitor: add SLA and Slack alerts on failure.

This repository includes code, templates, and a CLI that implements the agent flow. Use the release artifacts for a fast install, or build from source to customize templates.

Find releases and the installer here: https://github.com/geonamsoil/airflow-dag-agent/releases

Enjoy building safe, testable, and repeatable Airflow DAGs with the agent.