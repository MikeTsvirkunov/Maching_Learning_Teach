```mermaid
    classDiagram
        class ITeacher{
            +teach(X_train, y_train, colums_spec)
        }
        ..|>
        class NaiveBayesianTeacher{
            -NaiveBayesianClassificator classifier
            +teach(X_train, y_train, colums_spec)
        }

```