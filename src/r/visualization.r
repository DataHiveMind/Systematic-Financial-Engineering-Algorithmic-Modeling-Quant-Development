library(ggplot2)
library(reshape2)
library(shiny)
library(plotly)

# --- Time-Series Plot ---
plot_time_series <- function(df, date_col = "Date", value_col = "Close", title = "Time Series") {
  ggplot(df, aes_string(x = date_col, y = value_col)) +
    geom_line(color = "steelblue") +
    labs(title = title, x = "Date", y = value_col) +
    theme_minimal()
}

# --- Histogram ---
plot_histogram <- function(df, value_col = "returns", bins = 30, title = "Histogram") {
  ggplot(df, aes_string(x = value_col)) +
    geom_histogram(bins = bins, fill = "skyblue", color = "black", alpha = 0.7) +
    labs(title = title, x = value_col, y = "Frequency") +
    theme_minimal()
}

# --- Correlation Heatmap ---
plot_correlation_heatmap <- function(df, cols, title = "Correlation Matrix") {
  cor_mat <- cor(df[cols], use = "complete.obs")
  melted_cor <- melt(cor_mat)
  ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "red", high = "green", mid = "white", midpoint = 0) +
    labs(title = title, x = "", y = "") +
    theme_minimal()
}

# --- Risk Exposure Plot ---
plot_risk_exposure <- function(df, date_col = "Date", risk_col = "VaR", title = "Risk Exposure Over Time") {
  ggplot(df, aes_string(x = date_col, y = risk_col)) +
    geom_line(color = "firebrick") +
    labs(title = title, x = "Date", y = risk_col) +
    theme_minimal()
}

# --- Shiny Dashboard Skeleton ---
risk_dashboard <- function(df) {
  ui <- fluidPage(
    titlePanel("Financial Risk Dashboard"),
    sidebarLayout(
      sidebarPanel(
        selectInput("plot_type", "Select Plot:",
                    choices = c("Time Series", "Histogram", "Correlation Heatmap", "Risk Exposure"))
      ),
      mainPanel(
        plotlyOutput("main_plot")
      )
    )
  )
  server <- function(input, output) {
    output$main_plot <- renderPlotly({
      if (input$plot_type == "Time Series") {
        ggplotly(plot_time_series(df))
      } else if (input$plot_type == "Histogram") {
        ggplotly(plot_histogram(df))
      } else if (input$plot_type == "Correlation Heatmap") {
        ggplotly(plot_correlation_heatmap(df, names(df)))
      } else if (input$plot_type == "Risk Exposure") {
        ggplotly(plot_risk_exposure(df))
      }
    })
  }
  shinyApp(ui, server)
}

# --- Example Usage ---
# df <- read.csv("your_timeseries.csv")
# print(plot_time_series(df))
# print(plot_histogram(df))
# print(plot_correlation_heatmap(df, c("returns", "volatility", "momentum")))
# print(plot_risk_exposure(df, risk_col = "VaR"))
# risk_dashboard(df)