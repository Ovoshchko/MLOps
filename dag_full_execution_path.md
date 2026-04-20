# Full Execution Path: From Python Process to DAG Triggering

This document explains the complete flow from starting the Python process to DAG triggering in the SamazRec system.

## 1. System Architecture Overview

The SamazRec system is built on Apache Airflow for workflow orchestration. The execution path involves:

1. Airflow Scheduler Process
2. DAG Parsing and Registration
3. Schedule Evaluation
4. DAG Triggering
5. Task Execution

## 2. Starting Point: Airflow Scheduler

The entry point for the entire system is the Airflow scheduler, which is started as a service:

```bash
# Starting Airflow scheduler (typically done by infrastructure)
airflow scheduler

# Starting Airflow webserver (for UI access)
airflow webserver
```

## 3. DAG Discovery Process

### 3.1 Python Process Initialization
When Airflow starts, it initializes Python processes that:

1. Scan the `dags/` directory for Python files
2. Import each DAG file as a Python module
3. Execute the module code to create DAG objects

### 3.2 DAG File Execution
For `dags/universal_model_llm/dag_infer.py`:

```python
# 1. Import statements are executed
from datetime import datetime
from airflow.utils.task_group import TaskGroup
from amazme.dags.airflow.decorators import dag_factory
# ... other imports

# 2. DAG factory decorator is applied
@dag_factory(
    dag_id="universal_model_infer",
    schedule_interval=setup_schedule("30 2 * * *"),
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["common", "online"],
    params=MODEL_CONFIG,
)
def universal_model_infer():
    # DAG definition code is executed
    pass

# 3. DAG instance is created
# This is the crucial part - the DAG function is called to create the DAG object
universal_model_infer()  # <-- This creates the actual DAG instance
```

## 4. DAG Registration

When `universal_model_infer()` is called:

1. The `@dag_factory` decorator processes the DAG configuration
2. Creates an Airflow DAG object with the specified parameters
3. Registers the DAG with the Airflow scheduler
4. The DAG becomes available in the Airflow UI and scheduler

## 5. Schedule Evaluation

The Airflow scheduler continuously:

1. Checks the current time against each registered DAG's schedule
2. For `schedule_interval="30 2 * * *"`, it checks if it's 2:30 AM every day
3. When a match is found, creates a DAG run

## 6. DAG Triggering Process

### 6.1 Automatic Triggering
1. Scheduler detects schedule match
2. Creates a DAG run with a unique run_id
3. Sets the DAG run state to "running"
4. Begins executing tasks based on dependencies

### 6.2 Manual Triggering
1. User clicks "Trigger DAG" in Airflow UI
2. Airflow creates a DAG run with execution date
3. DAG execution begins immediately

## 7. Task Execution Flow

For the universal_model_infer DAG:

```mermaid
graph TD
    A[Airflow Scheduler Starts] --> B[DAG File Parsing]
    B --> C[@dag_factory Execution]
    C --> D[DAG Registration]
    D --> E[Schedule Evaluation]
    E --> F{Trigger Condition Met?}
    F -->|Yes| G[DAG Run Creation]
    F -->|Manual| H[DAG Run Creation]
    G --> I[Task Execution]
    H --> I
    I --> J[wait_for_data TaskGroup]
    I --> K[wait_for_um_train TaskGroup]
    I --> L[wait_for_collab_categories TaskGroup]
    I --> M[prepare_stocks TaskGroup]
    J --> N[prepare_items Task]
    K --> O[External Dependencies]
    L --> P[External Dependencies]
    M --> Q[Stock Preparation]
    N --> R[LLM Description Generation]
    O --> S[Embedding Generation]
    P --> T[Model Deployment]
    Q --> U[FAISS Index Creation]
    R --> S
    S --> U
    U --> T
```

## 8. Key Points About the "Missing" Main Function

Unlike traditional Python applications, Airflow DAGs don't need a `main()` function because:

1. **Declarative Approach**: DAGs declare workflow structure, not execution logic
2. **Scheduler-Driven**: Airflow scheduler handles execution timing
3. **Event-Based**: DAGs are triggered by events (time, API calls, etc.)
4. **Distributed Execution**: Tasks run on worker nodes, not a single process

## 9. Complete Execution Path Summary

1. **System Start**: Airflow services start (scheduler, webserver, workers)
2. **DAG Discovery**: Scheduler scans `dags/` directory
3. **Module Import**: Python imports DAG files
4. **DAG Creation**: `@dag_factory` decorator creates DAG objects
5. **DAG Registration**: DAGs registered with scheduler
6. **Schedule Monitoring**: Scheduler continuously checks schedules
7. **Trigger Event**: Schedule match or manual trigger
8. **DAG Run Creation**: Scheduler creates DAG run
9. **Task Scheduling**: Tasks scheduled based on dependencies
10. **Task Execution**: Workers execute individual tasks
11. **Completion**: DAG run marked as success/failed

This architecture allows for robust, scalable workflow orchestration without requiring explicit main functions or traditional program entry points.