
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
library(shiny)
# Define server function  
shinyServer(function(input, output,session) {
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
}) # server