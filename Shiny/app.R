packages = c("devtools","shiny","shinythemes","shinyFiles")

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
ui <- fluidPage(theme = shinytheme("cyborg"),
            navbarPage("Animal Identification",
                       tabsetPanel(type = "tabs",
                       tabPanel("Predict",value = "Predict",
                                mainPanel(
                                  h1("Prediction Results"),
                                  shinyDirButton('prediction_path_prefix', "Image directory", "Select the parent directory where images are stored"),
                                  actionButton("runSubmit","Submit",class="btn btn-primary"),
                                  textOutput("submitresult"),
                                  #tableOutput(outputId = 'table.output'),
                                ) # mainPanel
                       ),
                      tabPanel("Settings",value = "Settings",
                               sidebarPanel(
                                 #fileInput("prediction_data_info", "Choose the CSV file containing the image labels", multiple = FALSE, accept = c(".csv")),
                                 selectInput("prediction_type", label="Choose Prediction:",
                                             choices= list("Contains Animals"="empty_animal","Species Identification"="species_model","CFTEP"="CFTEP"), 
                                             selected = "empty_animal"),
                                 #shinyDirButton('prediction_model_dir', 'Models directory', title="Select the parent folder"),
                                 #selectInput("prediction_modelChoice", label="Choose Model:",
                                 #            choices= list("RESNET Model"="resnet","VGG Model"="vgg"), 
                                 #            selected = "Auto Selection"),
                               ))
                                  )
                       )
            )

# Define server function  
server <- function(input, output, session) {
  #first-time setup
  #- make file selection for some variables
  # base directory for fileChoose
  volumes = getVolumes()
  # python_loc
  dirname_python_loc <- reactive({parseDirPath(volumes, input$python_loc)})# Observe python_loc changes
  
#observe({
#  if(!is.null(dirname_python_loc)){
#    print(dirname_python_loc())
#    output$python_loc <- renderText(dirname_python_loc())
#  }
#})
# output$python_loc_print <- renderText({
#   paste0("setup(python_loc = '", normalizePath(dirname_python_loc()), "', ",
#          "r-reticulate = ", input$r_reticulate, ", ",
#          "gpu = ", input$gpu, ")","\n")
# })
# # run setup
# observeEvent(input$runSetup, {
#   showModal(modalDialog("Setting up environment... Dismiss anytime..."))
#   setup(
#     python_loc = gsub("\\\\", "/", paste0(normalizePath(dirname_python_loc()), "/")),
#     conda_loc = "auto",
#     #r_reticulate = promises::promise_resolve(input$r_reticulate),
#     gpu = input$gpu
#   )
#   showModal(modalDialog("Setup function complete."))
#   output$setupresult <- renderText("Setup Completed!")
# })
  
  #submit
  observeEvent(input$runSubmit, {
    showModal(modalDialog("Running model, this may take some time."))
    
    path_prefix <- parseDirPath(volumes, input$prediction_path_prefix)
    
    p_data_info <- input$prediction_data_info$datapath
    
    model_dir <- parseDirPath(volumes, input$prediction_model_dir)
    
    prediction_type <- input$prediction_type
    
    prediction_modelChoice <- input$prediction_modelChoice
    
    #withProgress(message = 'Generating data',
        classify(
             path_prefix = path_prefix, # path to where your images are stored
             data_info =  ,# path to csv containing file names and labels
             model_dir = model_dir, # path to the helper files that you downloaded in step 3, including the name of this directory (i.e., `MLWIC2_helper_files`). Check to make sure this directory includes files like arch.py and run.py. If not, look for another folder inside this folder called `MLWIC2_helper_files`
             python_loc = ,
             os = "Windows",
             save_predictions = "model_predictions.txt", # how you want to name the raw output file
             make_output = TRUE, # if TRUE, this will produce a csv with a more friendly output
             output_location = NULL,
             output_name = "MLWIC2_output.csv",
             num_cores = 4,
             log_dir = prediction_type,
             architecture = prediction_modelChoice,
             )
    #)
    #output$table.output <- renderTable({
    #  table <- read.csv()
    #  return(table)
    #})
    showModal(modalDialog("Complete"))
    })
  
  #predict
  shinyDirChoose(input, 'prediction_path_prefix', roots=volumes(), session=session)
  prediction_path_prefix <- reactive({parseDirPath(volumes, input$prediction_path_prefix)})
  
  prediction_data_info <- reactive({input$prediction_data_info}) 
  
  shinyDirChoose(input, 'prediction_model_dir', roots=volumes(), session=session)
  prediction_model_dir <- reactive({parseDirPath(volumes, input$prediction_model_dir)})
} # server

# Create Shiny object
shinyApp(ui = ui, server = server)