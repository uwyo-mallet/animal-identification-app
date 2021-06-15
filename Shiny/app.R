packages = c("devtools","shiny", "shinythemes","shinyFiles")

## Now load or install & load all
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
    }
    library(x, character.only = TRUE)
  }
)

if (!require('MLWIC2')) devtools::install_github("haniyeka/MLWIC2") 
library(MLWIC2)

setwd(".")
# Define UI
ui <- fluidPage(theme = shinytheme("flatly"),
            navbarPage("Animal Identification",
                       tabPanel("Predict",value = "Predict",
                                sidebarPanel(
                                  tags$h3("Predicting Inputs:"),
                                  fileInput("prediction_data_info","Image label CSV file",accept = c(".csv")),
                                  shinyDirButton('prediction_path_prefix', "Image directory", title="Select the parent directory where images are stored"),
                                  selectInput("prediction_Type", label="Choose Prediction:",
                                              choices= list("empty_animal"="empty_animal","species_model"="species_model"), 
                                              selected = "empty_animal"),
                                  selectInput("prediction_modelChoice", label="Choose Moel:",
                                              choices= list("Auto Selection"="AutoSelection","VGG Model"="VGGmodel", "RESNET Model"="Resnetmodel"), 
                                              selected = "Auto Selection"),
                                  shinyDirButton('prediction_model_dir', 'Models directory', title="Find and select the parent folder"),
                                  actionButton("submitbutton","Submit",class="btn btn-primary")
                                  
                                ), # sidebarPanel
                                mainPanel(
                                  h1("Prediction Results"),
                                  h4("Statistical Results"),
                                  verbatimTextOutput("statisticalResults"),
                                  textOutput("predict_command_print")
                                  
                                ) # mainPanel
                                
                       ),
                       tabPanel("Retrain",value = "Retrain", "This panel is intentionally left blank"),
                       navbarMenu("Setting",
                                  tabPanel("First-time Setup",value = "Setup",
                                           sidebarPanel(
                                             shinyDirButton('python_loc', "Python location", title="Select the location of Python. It should be under Anaconda. Just select the folder where it resides in the top half of the menu and press `Select`"),
                                             selectInput("r_reticulate", "Have you already installed packages in aconda environment called `r-reticulate` that you want to keep?
                                                         If you don't know, click `No`",
                                                         choices = c(
                                                           "No" = FALSE,
                                                           "Yes" = TRUE
                                                         )
                                             ),
                                             selectInput("gpu", "Do you have a GPU on your machine that you are planning to use?
                                                           If you don't know, click `No`",
                                                         choices = c(
                                                           "No" = FALSE,
                                                           "Yes" = TRUE
                                                         )
                                             ),
                                             actionButton("runSetup", "Run Setup Function")
                                           ),
                                           # Main panel for displaying outputs ----
                                           mainPanel(
                                             helpText("Use this intallation just for the first time. This will install packages and setup the conda environment!"),
                                             textOutput("python_loc_print"),
                                             textOutput("setupresult")
                                           )
                                  )
                       )
            )
  )

# Define server function  
server <- function(input, output,session) {
  #first-time setup
  #- make file selection for some variables
  # base directory for fileChoose
  volumes = getVolumes()
  # python_loc
  shinyDirChoose(input, 'python_loc', roots=volumes(), session=session)
  dirname_python_loc <- reactive({parseDirPath(volumes, input$python_loc)})# Observe python_loc changes
  observe({
    if(!is.null(dirname_python_loc)){
      print(dirname_python_loc())
      output$python_loc <- renderText(dirname_python_loc())
    }
  })
  output$python_loc_print <- renderText({
    paste0("setup(python_loc = '", normalizePath(dirname_python_loc()), "', ",
           "r-reticulate = ", input$r_reticulate, ", ",
           "gpu = ", input$gpu, ")","\n")
  })
  #- run classify
  observeEvent(input$runSetup, {
    showModal(modalDialog("Setting up environment... Dismiss anytime..."))
    setup(
      python_loc = gsub("\\\\", "/", paste0(normalizePath(dirname_python_loc()), "/")),
      conda_loc = "auto",
      #r_reticulate = promises::promise_resolve(input$r_reticulate),
      gpu = input$gpu
    )
    showModal(modalDialog("Setup function complete."))
    output$setupresult <- renderText("Setup Completed!")
  })
  
  
  #predict
  
  observeEvent(
    input$predictionType,
    if(input$predictionType == "species_model"){
      updateSelectInput(session, "prediction_modelChoice",choices = list("RESNET Model"="Resnetmodel"))
    }
    else{
      updateSelectInput(session, "prediction_modelChoice",choices = list("Auto Selection"="AutoSelection","VGG Model"="VGGmodel", "RESNET Model"="Resnetmodel"))
    })
  
  shinyDirChoose(input, 'prediction_path_prefix', roots=volumes(), session=session)
  prediction_path_prefix <- reactive({parseDirPath(volumes, input$prediction_path_prefix)})
  observe({
    if(!is.null(prediction_path_prefix)){
      print(prediction_path_prefix())
      output$predict_command_print <- renderText(prediction_path_prefix())
    }
  })
  shinyDirChoose(input, 'prediction_model_dir', roots=volumes(), session=session)
  prediction_model_dir <- reactive({parseDirPath(volumes, input$prediction_model_dir)})
  observe({
    if(!is.null(prediction_model_dir)){
      print(prediction_model_dir())
      output$predict_command_print <- renderText(prediction_model_dir())
    }
  })
  prediction_data_info <- reactive({input$prediction_data_info}) 
  observe({
    if(!is.null(prediction_data_info)){
      print(prediction_data_info())
      output$predict_command_print <- renderDataTable(prediction_data_info())
    }
  })
} # server

# Create Shiny object
shinyApp(ui = ui, server = server)